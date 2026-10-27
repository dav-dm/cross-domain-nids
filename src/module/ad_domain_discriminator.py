import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, p_drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(p_drop),
            nn.Linear(128, 64),        
            nn.BatchNorm1d(64),  
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),       
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector:
    """Wraapper skelearn-style interface for the Autoencoder model."""
    # TODO: Rewrite in pytorch
    def __init__(
        self, input_dim=120, latent_dim=32, epochs=50, batch_size=64, lr=1e-3, device=None, 
        seed=0, quantile=0.95):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed
        self.quantile = quantile
        self._model = Autoencoder(input_dim, latent_dim).to(self.device)
        self.threshold_ = None

    @staticmethod
    def _to_tensor(x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to float32 tensor."""
        return torch.from_numpy(x.astype(np.float32))

    def fit(self, X: np.ndarray):
        """Train the autoencoder and compute the reconstruction-error threshold."""
        torch.manual_seed(self.seed)
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss(reduction='none')

        X_tensor = self._to_tensor(X)
        loader = torch.utils.data.DataLoader(
            X_tensor,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for _ in tqdm(range(self.epochs), desc='[training Autoencoder]'):
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch).mean()
                loss.backward()
                optimizer.step()

        # Determine threshold on training set
        self._model.eval()
        with torch.no_grad():
            recon = self._model(X_tensor.to(self.device)).cpu()
        errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
        self.threshold_ = np.quantile(errors, self.quantile)
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error for each sample."""
        self._model.eval()
        X_tensor = self._to_tensor(X).to(self.device)
        with torch.no_grad():
            recon = self._model(X_tensor).cpu()
        errs = ((recon - X_tensor.cpu()) ** 2).mean(dim=1).numpy()
        return errs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for normal, -1 for anomaly."""
        errs = self._reconstruction_error(X)
        return np.where(errs <= self.threshold_, 1, -1)