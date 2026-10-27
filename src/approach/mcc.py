import sys
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from approach.dl_module import DLModule
from data.infinite_data_iterator import InfiniteDataIterator
from module.loss import MinimumClassConfusionLoss
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class MCC(DLModule):
    """
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcc.py)
    MCC is a class that implements the approach described in "Minimum Class Confusion for Versatile Domain Adaptation" for domain adaptation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
                
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mcc_t = kwargs.get('mcc_t', cf['mcc_t'])
        self.alpha = kwargs.get('mcc_alpha', cf['mcc_alpha'])
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        self.iter_per_epoch = kwargs.get('iter_per_epoch', cf['iter_per_epoch'])
        
        # Minimum class confusion loss function
        self.mcc_loss = MinimumClassConfusionLoss(T=self.mcc_t)
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--mcc-t', type=float, default=cf['mcc_t'])
        parser.add_argument('--mcc-alpha', type=float, default=cf['mcc_alpha'])
        parser.add_argument('--adapt-lr', type=float, default=cf['adapt_lr'])
        parser.add_argument('--adapt-epochs', type=int, default=cf['adapt_epochs'])
        parser.add_argument('--iter-per-epoch', type=int, default=cf['iter_per_epoch'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
    
    
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
        

    def _adapt(self, adapt_dataloader, val_dataloader, train_dataloader):
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass').to(self.device)
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device)
        
        # Infinite iterators
        src_dataloader = InfiniteDataIterator(train_dataloader, device=self.device) 
        trg_dataloader = InfiniteDataIterator(adapt_dataloader, device=self.device) 
        
        self.lr = self.adapt_lr
        self.configure_optimizers()
        
        # Adaptation loop       
        for epoch in range(self.adapt_epochs):
            self.net.train()
            
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
            for i in adapt_loop:
                # Get batches
                batch_x_s, batch_y_s = next(src_dataloader)
                batch_x_t, _ = next(trg_dataloader)
                batch_x_s, batch_y_s = batch_x_s.to(self.device), batch_y_s.to(self.device).long()
                batch_x_t = batch_x_t.to(self.device)
                
                # Compute logits
                batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
                logits = self.net(batch_x)
                logits_s, logits_t = torch.split(
                    logits, [batch_x_s.size(0), batch_x_t.size(0)], dim=0)
                
                # Compute losses
                ce_loss = self.ce_loss(logits_s, batch_y_s)
                mcc_loss = self.mcc_loss(logits_t)
                loss = ce_loss + mcc_loss * self.alpha
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                # Metrics
                preds_s = torch.argmax(logits_s, dim=1)
                all_labels.append(batch_y_s)
                all_preds.append(preds_s)
                
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
            