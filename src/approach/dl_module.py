import sys
import torch
import importlib
from tqdm import tqdm
from argparse import ArgumentParser
from torch import optim
from torchmetrics import Accuracy, F1Score

from util.config import load_config
from network.network_factory import build_network

disable_tqdm = not sys.stdout.isatty()
dl_approaches = {
    'baseline' : 'Baseline',
    'adda' : 'ADDA',
    'mcc' : 'MCC',
    'cdan' : 'CDAN',
    'mcd' : 'MCD',
    'bsp' : 'BSP',
    'icon' : 'ICON',
}
dl_unsup_approaches = [
    'adda', 'mcc', 'cdan', 'mcd', 'bsp', 'icon'
]


class DLModule:
    
    def __init__(self, datamodule=None, callbacks=None, **kwargs):
        cf = load_config()
        
        gpu = kwargs.get('gpu', cf['gpu'])
        self.device = torch.device('cuda') if gpu and torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        
        self.net = build_network(**kwargs).to(self.device)
        self.net.summarize_module()
        
        self.datamodule = datamodule
        self.num_classes = kwargs.get('num_classes', cf['num_classes'])
                
        self.lr = kwargs.get('lr', cf['lr'])
        self.lr_strat = kwargs.get('lr_strat', cf['lr_strat'])
        self.optimizer_name = kwargs.get('optimizer', cf['optimizer'])
        self.sch_monitor = kwargs.get('sch_monitor', cf['sch_monitor'])
        
        self.max_epochs = kwargs.get('max_epochs', cf['max_epochs'])
        self.min_epochs = kwargs.get('min_epochs', cf['min_epochs'])
            
        self.task = 'src'    
        self.phase = None
        self.outputs = None
        self.checkpoint_path = None
        
        self.callbacks = callbacks if callbacks is not None else []
        
        self.configure_optimizers()
        
        
    @staticmethod
    def get_approach(appr_name, **kwargs):
        Approach = getattr(importlib.import_module(
            name=f'approach.{appr_name}'), dl_approaches[appr_name])
        return Approach(**kwargs)
    
    
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--lr', type=float, default=cf['lr'])
        parser.add_argument('--lr-strat', type=str, default=cf['lr_strat'])
        parser.add_argument('--sch-monitor', type=str, default=cf['sch_monitor'])
        parser.add_argument('--max-epochs', type=int, default=cf['max_epochs'])
        parser.add_argument('--min-epochs', type=int, default=cf['min_epochs'])
        parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], 
                            default=cf['optimizer'])
        return parser
    
    #------------------------------------
    # TRAIN, VALIDATION, TEST, ADAPTATION
    #------------------------------------
    
    def fit(self):
        self.phase = 'train'

        for cb in self.callbacks: 
            cb.on_fit_start(self)
            
        train_dataloader = self.datamodule.get_train_data()
        val_dataloader = self.datamodule.get_val_data()
        self._fit(train_dataloader, val_dataloader)

        for cb in self.callbacks:
            cb.on_fit_end(self)
        
    def test(self):
        self.phase = 'test'
        
        for cb in self.callbacks:
            cb.on_test_start(self)
            
        test_dataloader = self.datamodule.get_test_data()
        self.outputs = self._predict(test_dataloader)
        
        for cb in self.callbacks:
            cb.on_test_end(self)
        
    def validate(self):
        self.phase = 'val'
        
        for cb in self.callbacks:
            cb.on_validation_start(self)
            
        val_dataloader = self.datamodule.get_val_data()
        self.outputs = self._predict(val_dataloader)
        
        for cb in self.callbacks:
            cb.on_validation_end(self)  
            
    def adapt(self, **kwargs):
        self.phase = 'train'

        for cb in self.callbacks: 
            cb.on_adaptation_start(self)
            
        # Adaptation (on trg dataset) dataloder set by the trainer
        adapt_dataloader = self.datamodule.get_adapt_data() 
        val_dataloader = self.datamodule.get_val_data()
        self._adapt(adapt_dataloader, val_dataloader, **kwargs)

        for cb in self.callbacks:
            cb.on_adaptation_end(self) 
            
    #-------------
    # TEMPLATE FIT
    #-------------
    
    def _fit(self, train_dataloader, val_dataloader):
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass').to(self.device)
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device)
        
        for epoch in range(self.max_epochs):
            self.net.train()
            self._set_train_mode()

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
            train_loop = tqdm(
                train_dataloader, desc=f'Ep[{epoch+1}/{self.max_epochs}]',  
                postfix=postfix, leave=False, disable=disable_tqdm
            )
            for batch_x, batch_y in train_loop:
                # Move data on self.device
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
                # Forward pass and Loss
                loss, logits = self._fit_step(batch_x, batch_y.long())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Metrics
                preds = torch.argmax(logits, dim=1)
                all_labels.append(batch_y)
                all_preds.append(preds)
                
            # Validation on fit epoch end
            self.epoch_outputs = self._predict(val_dataloader, on_train_epoch_end=True)
            val_score = self.epoch_outputs[self.sch_monitor]
            
            self.epoch_outputs['train_loss'] = running_loss / len(train_dataloader)
            self.epoch_outputs['train_accuracy'] = accuracy(
                torch.cat(all_preds), torch.cat(all_labels)).item()
            self.epoch_outputs['train_f1_score_macro'] = f1_score(
                torch.cat(all_preds), torch.cat(all_labels)).item()

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            if self.should_stop:
                break  # Early stopping

            self.run_scheduler_step(monitor_value=val_score, epoch=epoch + 1)
            
    #-----------------
    # TEMPLATE PREDICT
    #-----------------        
    
    def _predict(self, dataloader, on_train_epoch_end=False):
        self.net.eval()
        self._set_eval_mode()
        
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass').to(self.device)
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device)
        
        all_labels, all_preds, all_logits = [], [], []
        running_loss = 0.0
        
        desc = '[val]' if on_train_epoch_end else f'[{self.phase}]'
        with torch.no_grad():
            
            predict_loop = tqdm(
                dataloader, desc=desc, leave=not self.phase=='train', disable=disable_tqdm
            )
            for batch_x, batch_y in predict_loop:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                loss, logits = self._predict_step(batch_x, batch_y.long())
            
                preds = torch.argmax(logits, dim=1)
                running_loss += loss.item() * batch_y.shape[0]
                
                all_labels.append(batch_y)
                all_preds.append(preds)
                all_logits.append(logits)
                
        labels = torch.cat(all_labels)
        preds = torch.cat(all_preds)
        logits = torch.cat(all_logits)
        
        eval_loss = running_loss / labels.shape[0] if labels.shape[0] > 0 else 0.0
        
        return {
            'accuracy' : accuracy(preds, labels).item(),
            'f1_score_macro' : f1_score(preds, labels).item(),
            'loss' : eval_loss,
            'labels': labels.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy(),
            'logits': logits.detach().cpu().numpy(),
        }
            
    #------------------------
    # SCHEDULER AND OPTIMIZER
    #------------------------
    
    def configure_optimizers(self, params=None):
        cf = load_config()
        
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(params or self.net.parameters(), lr=self.lr)
            
        elif self.optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(params or self.net.parameters(), lr=self.lr)
            
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                params or self.net.parameters(), lr=self.lr, 
                momentum=cf['sgd_params']['momentum'], 
                weight_decay=cf['sgd_params']['weight_decay'], 
                nesterov=cf['sgd_params']['nesterov']
            )
        else:
            raise ValueError('Optimizer not implemented')
        
        if self.lr_strat == 'none':
            self.scheduler = None

        elif self.lr_strat == 'lrop':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cf['lrop_mode'],
                factor=cf['lrop_factor'],
                patience=cf['lrop_patience'],
            )

        elif self.lr_strat == 'cawr':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cf['cawr_t0'],
                T_mult=cf['cawr_t_mult'],
                eta_min=float(cf['cawr_eta_min'])
            )
        else:
            raise ValueError('Scheduler not implemented')
        
        print(f'[Optimizer] {self.optimizer_name} with {self.lr} and {self.lr_strat} lr scheduler')   
            
    
    def run_scheduler_step(self, monitor_value, epoch=None):
        if self.scheduler is None:
            return
        if self.lr_strat == 'lrop':
            self.scheduler.step(monitor_value)
        elif self.lr_strat == 'cawr':
            self.scheduler.step(epoch)
        else:
            raise ValueError('Scheduler not implemented')
        
        if hasattr(self.scheduler, 'get_last_lr'):
            new_lr = self.scheduler.get_last_lr()[0]
            print(f'[Scheduler][Ep{epoch}] Current LR after step: {new_lr:.6f}')
        
    #------------------------
    # CHECKPOINTING
    #------------------------
    
    def load_checkpoint(self, path):
        """
        Loads the checkpoint from a specified file path.
        """
        if path is None:
            print('WARNING: checkpoint path is None, using current state.')
            return
        if torch.cuda.is_available():
            checkpoint = torch.load(path, weights_only=True)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        self.net.load_state_dict(checkpoint['net_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            if 'parameter group' in str(e):
                print('WARNING: Optimizer state dict not loaded. Checkpoint may be incompatible.')
            else:
                raise e
        # for name, param in self.net.named_parameters():
        #     print(name, param.device, torch.sum(param).item())
        
    def save_checkpoint(self, path):
        """
        Saves the current state of the network and optimizer to a file.
        """
        # for name, param in self.net.named_parameters():
        #     print(name, param.device, torch.sum(param).item()) 
        self.checkpoint_path = path       
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)
        return self.checkpoint_path
    
    #------------------------
    # UTILITY
    #------------------------    

    def _set_train_mode(self):
        """
        This function can be overriden by approaches that need to set 
        in train mode additional nn modules
        """
        pass
    
    def _set_eval_mode(self):
        """
        This function can be overriden by approaches that need to set 
        in eval mode additional nn modules
        """
        pass