import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from approach.ml_module import MLModule
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()

class _AE3Net(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        n = n_features
        self.encoder = nn.Sequential(
            nn.Linear(n, round(0.83 * n)), nn.ReLU(),
            nn.Linear(round(0.83 * n), round(0.66 * n)), nn.ReLU(),
            nn.Linear(round(0.66 * n), round(0.50 * n)), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(round(0.50 * n), round(0.66 * n)), nn.ReLU(),
            nn.Linear(round(0.66 * n), round(0.83 * n)), nn.ReLU(),
            nn.Linear(round(0.83 * n), n), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DAE3(MLModule):
    """
    DAE-3 deep autoencoder for anomaly detection.
    From "Network anomaly detection methods in IoT environments via deep learning:
    A Fair comparison of performance and robustness"
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        gpu = kwargs.get('gpu', cf['gpu'])
        self.device = torch.device('cuda') if gpu and torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')

        self.max_epochs = kwargs.get('dae3_max_epochs', cf['dae3_max_epochs'])
        self.lr = kwargs.get('dae3_lr', cf['dae3_lr'])
        self.momentum = kwargs.get('dae3_momentum', cf['dae3_momentum'])
        self.batch_size = kwargs.get('dae3_batch_size', cf['dae3_batch_size'])
        self.patience = kwargs.get('dae3_patience', cf['dae3_patience'])
        self.k_threshold = kwargs.get('dae3_k_threshold', cf['dae3_k_threshold'])

        self.net = None
        self.threshold = None

    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--dae3-max-epochs', type=int, default=cf['dae3_max_epochs'])
        parser.add_argument('--dae3-lr', type=float, default=cf['dae3_lr'])
        parser.add_argument('--dae3-momentum', type=float, default=cf['dae3_momentum'])
        parser.add_argument('--dae3-batch-size', type=int, default=cf['dae3_batch_size'])
        parser.add_argument('--dae3-patience', type=int, default=cf['dae3_patience'])
        parser.add_argument('--dae3-k-threshold', type=int, default=cf['dae3_k_threshold'])
        return parser


    def _fit(self, data, _):
        n_features = data.shape[1]
        self.net = _AE3Net(n_features).to(self.device)

        x_val, _ = self.datamodule.get_val_data()
        x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        x_train = torch.tensor(data, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(x_train), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        criterion = nn.MSELoss()

        best, wait = float('inf'), 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.net.train()
            running_loss = 0.0

            postfix = {
                'trn_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
            } if epoch > 0 else {}
            train_loop = tqdm(
                loader, leave=False, desc=f'Ep[{epoch+1}/{self.max_epochs}]', 
                postfix=postfix, disable=disable_tqdm
            )
            for batch_idx, (x_batch,) in enumerate(train_loop, start=1):
                x_batch = x_batch.to(self.device)
                loss = criterion(self.net(x_batch), x_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_train_loss = running_loss / batch_idx

            self.net.eval()
            with torch.no_grad():
                val_loss = criterion(self.net(x_val), x_val).item()

            if val_loss < best - 1e-4:
                best = val_loss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            # Load best model state
            self.net.load_state_dict(best_state)
            self.net.to(self.device)

        scores = self._rmse(x_train)
        self.threshold = (scores.mean() + self.k_threshold * scores.std()).item()
        print(f'DAE3 threshold set to: {self.threshold:.4f}')


    def _predict(self, data, labels):
        x_test = torch.tensor(data, dtype=torch.float32)

        scores = self._rmse(x_test.to(self.device)).detach().cpu().numpy()
        preds = (scores > self.threshold).astype(int)
        
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'labels': labels,
            'preds': preds
        }

    @torch.no_grad()
    def _rmse(self, x):
        self.net.eval()
        x_hat = self.net(x)
        return torch.sqrt(torch.mean((x - x_hat) ** 2, dim=1))