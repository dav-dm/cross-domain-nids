import sys
import torch
from torch import optim, nn
from tqdm import tqdm

from approach.dl_module import DLModule
from data.infinite_data_iterator import InfiniteDataIterator
from module.head import FullyConnected
from module.loss import ClassifierDiscrepancy, MCDEntropy
from torchmetrics import Accuracy, F1Score
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class MCD(DLModule):
    """
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcd.py)
    MCD is a class that implements the approach described in "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation" for domain adaptation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.num_classes = kwargs.get('num_classes', cf['num_classes'])

        # MCD hyperparameters
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        self.iter_per_epoch = kwargs.get('iter_per_epoch', cf['iter_per_epoch'])
        self.trade_off = kwargs.get('mcd_trade_off', cf['mcd_trade_off'])
        self.trade_off_entropy = kwargs.get('mcd_trade_off_entropy', cf['mcd_trade_off_entropy'])
        self.num_k = kwargs.get('mcd_num_k', cf['mcd_num_k'])

        # Two classifier heads
        self.head1 = FullyConnected(
            in_features=self.net.out_features_size,
            num_classes=self.num_classes
        ).to(self.device)
        self.head2 = FullyConnected(
            in_features=self.net.out_features_size,
            num_classes=self.num_classes
        ).to(self.device)

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.disc_loss = ClassifierDiscrepancy().to(self.device)
        self.entropy = MCDEntropy().to(self.device)

        # Override the default optimizer created in DLModule.__init__
        # so that fit() updates net + heads (DLModule.fit uses self.optimizer).
        self._configure_fit_optimizer()

    def _configure_fit_optimizer(self):
        """
        Make DLModule.fit() train net + head1 + head2 by rebuilding self.optimizer
        using DLModule.configure_optimizers(params=...).
        """
        params = list(self.net.parameters()) + list(self.head1.parameters()) + list(self.head2.parameters())
        self.configure_optimizers(params=params)
        
    def _set_train_mode(self):
        self.head1.train()
        self.head2.train()
        
    def _set_eval_mode(self):
        self.head1.eval()
        self.head2.eval()

    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--adapt-lr', type=float, default=cf['adapt_lr'])
        parser.add_argument('--adapt-epochs', type=int, default=cf['adapt_epochs'])
        parser.add_argument('--iter-per-epoch', type=int, default=cf['iter_per_epoch'])
        parser.add_argument('--mcd-trade-off', type=float, default=cf['mcd_trade_off'])
        parser.add_argument('--mcd-trade-off-entropy', type=float, default=cf['mcd_trade_off_entropy'])
        parser.add_argument('--mcd-num-k', type=int, default=cf['mcd_num_k'])
        return parser
    

    def _fit_step(self, batch_x, batch_y):
        _, feats = self.net(batch_x, return_feat=True)
        out1 = self.head1(feats)
        out2 = self.head2(feats)

        loss = self.ce_loss(out1, batch_y) + self.ce_loss(out2, batch_y)
        logits_avg = (out1 + out2) / 2.0
        return loss, logits_avg
    

    def _predict_step(self, batch_x, batch_y):
        _, feats = self.net(batch_x, return_feat=True)
        out1 = self.head1(feats)
        out2 = self.head2(feats)

        logits_avg = (out1 + out2) / 2.0
        loss = self.ce_loss(logits_avg, batch_y)
        return loss, logits_avg


    def _adapt(self, adapt_dataloader, val_dataloader, train_dataloader):
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass').to(self.device)
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device)
        
        src_dataloader = InfiniteDataIterator(train_dataloader, device=self.device)
        trg_dataloader = InfiniteDataIterator(adapt_dataloader, device=self.device)

        # Two optimizers: one for net + heads (step A), one for heads only (step B)
        optimizer_g = optim.Adam(self.net.parameters(), lr=self.adapt_lr)
        optimizer_f = optim.Adam(
            list(self.head1.parameters()) + list(self.head2.parameters()),
            lr=self.adapt_lr
        )

        # Adaptation loop
        for epoch in range(self.adapt_epochs):
            self.net.train()
            self.head1.train()
            self.head2.train()

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)
                
            running_loss = 0.0  
            all_labels, all_preds = [], []

            postfix = {
                'trn loss': f'{self.epoch_outputs["train_loss"]:.4f}', 
                'trn acc':  f'{self.epoch_outputs["train_accuracy"]:.4f}',
                'trn f1':   f'{self.epoch_outputs["train_f1_score_macro"]:.4f}',
                f'val {self.sch_monitor}': f'{val_score:.4f}'
            } if epoch > 0 else {} 
            adapt_loop = tqdm(
                range(self.iter_per_epoch), desc=f'Ep[{epoch+1}/{self.adapt_epochs}]',  
                postfix=postfix, leave=False, disable=disable_tqdm
            )
            for _ in adapt_loop:
                # Get batches
                batch_x_s, batch_y_s = next(src_dataloader)
                batch_x_t, _ = next(trg_dataloader)
                batch_x_s, batch_y_s = batch_x_s.to(self.device), batch_y_s.to(self.device).long()
                batch_x_t = batch_x_t.to(self.device)
                
                batch_x = torch.cat([batch_x_s, batch_x_t], dim=0)

                # Step A: train G + F  (min CE on source + entropy on target)
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                _, batch_emb = self.net(batch_x, return_feat=True)
                batch_emb_s, batch_emb_t = torch.split(
                    batch_emb, [batch_x_s.size(0), batch_x_t.size(0)], dim=0
                )
                batch_logits1_s = self.head1(batch_emb_s)
                batch_logits2_s = self.head2(batch_emb_s)
                batch_logits1_t = self.head1(batch_emb_t)
                batch_logits2_t = self.head2(batch_emb_t)

                loss_a = (
                    self.ce_loss(batch_logits1_s, batch_y_s)
                    + self.ce_loss(batch_logits2_s, batch_y_s)
                    + self.trade_off_entropy * (self.entropy(batch_logits1_t) + self.entropy(batch_logits2_t))
                )
                loss_a.backward()
                optimizer_g.step()
                optimizer_f.step()

                # Step B: train F (maximize CE on source + entropy on target + discrepancy on target)
                optimizer_f.zero_grad()
                
                with torch.no_grad():
                    _, batch_emb = self.net(batch_x, return_feat=True)
                    
                batch_emb = batch_emb.detach()
                batch_emb_s, batch_emb_t = torch.split(
                    batch_emb, [batch_x_s.size(0), batch_x_t.size(0)], dim=0
                )
                batch_logits1_s = self.head1(batch_emb_s)
                batch_logits2_s = self.head2(batch_emb_s)
                batch_logits1_t = self.head1(batch_emb_t)
                batch_logits2_t = self.head2(batch_emb_t)

                loss_b = (
                    self.ce_loss(batch_logits1_s, batch_y_s)
                    + self.ce_loss(batch_logits2_s, batch_y_s)
                    + self.trade_off_entropy * (self.entropy(batch_logits1_t) + self.entropy(batch_logits2_t))
                    - self.trade_off * self.disc_loss(batch_logits1_t, batch_logits2_t)
                )
                loss_b.backward()
                optimizer_f.step()

                # Step C: train G (min discrepancy on target)
                for _ in range(self.num_k):
                    optimizer_g.zero_grad()

                    _, batch_emb = self.net(batch_x, return_feat=True)
                    batch_emb_t = batch_emb[batch_x_s.size(0):]

                    batch_logits1_t = self.head1(batch_emb_t)
                    batch_logits2_t = self.head2(batch_emb_t)

                    loss_c = self.trade_off * self.disc_loss(batch_logits1_t, batch_logits2_t)
                    loss_c.backward()
                    optimizer_g.step()
                
                running_loss += loss_b.item()
                
                all_labels.append(batch_y_s)
                all_preds.append(torch.argmax((batch_logits1_s + batch_logits2_s) / 2.0, dim=1))

            # Validation on adapt epoch end
            self.epoch_outputs = self._predict(val_dataloader, on_train_epoch_end=True)
            val_score = self.epoch_outputs[self.sch_monitor]

            self.epoch_outputs['train_loss'] = running_loss / self.iter_per_epoch
            self.epoch_outputs['train_accuracy'] = accuracy(
                torch.cat(all_preds), torch.cat(all_labels)).item()
            self.epoch_outputs['train_f1_score_macro'] = f1_score(
                torch.cat(all_preds), torch.cat(all_labels)).item()
            
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            if self.should_stop:
                break  # Early stopping
            
            self.run_scheduler_step(monitor_value=val_score, epoch=epoch + 1)

    # ------------------------
    # CHECKPOINTING (override)
    # ------------------------

    def load_checkpoint(self, path):
        """
        Load net + heads (+ optimizer if present).
        """
        if path is None:
            print('WARNING: checkpoint path is None, using current state.')
            return
        if torch.cuda.is_available():
            checkpoint = torch.load(path, weights_only=True)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)

        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.head1.load_state_dict(checkpoint['head1_state_dict'])
        self.head2.load_state_dict(checkpoint['head2_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            if 'parameter group' in str(e):
                print('WARNING: Optimizer state dict not loaded. Checkpoint may be incompatible.')
            else:
                raise e
            
    def save_checkpoint(self, path):
        """
        Save net + heads (+ optimizer from fit, if you care).
        Keeps compatibility with DirectoryManager usage in DLModule.
        """
        self.checkpoint_path = path 
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'head1_state_dict': self.head1.state_dict(),
            'head2_state_dict': self.head2.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)
        return self.checkpoint_path