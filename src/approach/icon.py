import sys
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from approach.dl_module import DLModule
from data.infinite_data_iterator import InfiniteDataIterator
from module.head import FullyConnected
from module.loss import ClusterLoss, TsallisEntropy, PairwiseBCE, pair_enum
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class ICON(DLModule):
    """
    [[Link to Source Code]]()
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        # Adaptation hyperparameters
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        self.iter_per_epoch = kwargs.get('iter_per_epoch', cf['iter_per_epoch'])
        self.threshold = kwargs.get('icon_threshold', cf['icon_threshold'])
        self.back_cluster_ep = kwargs.get('back_cluster_start_epoch', cf['back_cluster_start_epoch'])
        
        # Init loss function parameters
        cluster_bce_type = kwargs.get('cluster_bce_type', cf['cluster_bce_type'])
        cosine_threshold = kwargs.get('cosine_threshold', cf['cosine_threshold'])
        topk = kwargs.get('topk', cf['topk'])
        tsallis_temperature = kwargs.get('tsallis_temperature', cf['tsallis_temperature'])
        tsallis_alpha = kwargs.get('tsallis_alpha', cf['tsallis_alpha'])
        
        # Init losses
        self.cluster_loss = ClusterLoss(
            num_classes=self.num_classes, bce_type=cluster_bce_type, 
            cosine_threshold=cosine_threshold, topk=topk
        )
        self.ts_loss = TsallisEntropy(temperature=tsallis_temperature, alpha=tsallis_alpha) 
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.st_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = PairwiseBCE()
        
        # Loss weights
        self.w_transfer = kwargs.get('w_transfer', cf['w_transfer'])
        self.w_self_training = kwargs.get('w_self_training', cf['w_self_training'])
        self.w_inv = kwargs.get('w_inv', cf['w_inv'])
        
        # Init additional heads
        self.cluster_head = FullyConnected(self.net.out_features_size, self.num_classes).to(self.device)
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--adapt-lr', type=float, default=cf['adapt_lr'])
        parser.add_argument('--adapt-epochs', type=int, default=cf['adapt_epochs'])
        parser.add_argument('--iter-per-epoch', type=int, default=cf['iter_per_epoch'])
        parser.add_argument('--cluster-bce-type', type=str, default=cf['cluster_bce_type'])
        parser.add_argument('--cosine-threshold', type=float, default=cf['cosine_threshold'])
        parser.add_argument('--topk', type=int, default=cf['topk'])
        parser.add_argument('--icon-threshold', type=float, default=cf['icon_threshold'])
        parser.add_argument('--tsallis-temperature', type=float, default=cf['tsallis_temperature'])
        parser.add_argument('--tsallis-alpha', type=float, default=cf['tsallis_alpha'])
        parser.add_argument('--w-transfer', type=float, default=cf['w_transfer'])
        parser.add_argument('--w-self-training', type=float, default=cf['w_self_training'])
        parser.add_argument('--w-inv', type=float, default=cf['w_inv'])
        parser.add_argument('--back-cluster-start-epoch', type=int, 
                            default=cf['back_cluster_start_epoch'])
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
        
        # ICON adaptation assumes that only the backbone is pre-trained.
        # During adaptation, the head is randomly initialized
        self.net.set_head(FullyConnected(self.net.out_features_size, self.num_classes).to(self.device))
        self.lr = self.adapt_lr
        params = list(self.net.parameters()) + list(self.cluster_head.parameters())
        self.configure_optimizers(params=params)
        
        # Adaptation loop       
        for epoch in range(self.adapt_epochs):
            self.net.train()
            
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)
                
            running_loss = 0.0
            all_labels, all_preds = [], []
            
            back_cluster = epoch >= self.back_cluster_ep # When to start using the backward clustering loss
                        
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
                
                batch = self._prepare_batches(src_dataloader, trg_dataloader)
                
                x_w = torch.cat([batch['x_s_w'], batch['x_t_w']], dim=0) # Weakly augmented inputs
                x_s = torch.cat([batch['x_s_s'], batch['x_t_s']], dim=0) # Strongly augmented inputs
                 
                y_w, feat_w = self.net(x_w, return_feat=True) # Logits and features for weakly augmented inputs
                y_s, feat_s = self.net(x_s, return_feat=True) # Logits and features for strongly augmented inputs
                
                # Get nograd versions by detaching features and passing through network head
                feat_w_ng = feat_w.detach()
                feat_s_ng = feat_s.detach()
                
                # Get nograd logits
                y_w_ng = self.net.head(feat_w_ng)
                y_s_ng = self.net.head(feat_s_ng)
                
                # Cluster head outputs with grad
                y_cls_w = self.cluster_head(feat_w)
                y_cls_s = self.cluster_head(feat_s)
                
                # Cluster head outputs without grad
                y_cls_w_ng = self.cluster_head(feat_w_ng)
                y_cls_s_ng = self.cluster_head(feat_s_ng)
                
                batch_size_s = batch['x_s_w'].size(0) # Source batch size
                batch_size_t = batch['x_t_w'].size(0) # Target batch size
                
                # Split logits into source and target
                y = self._split_logits(
                    outputs={
                        'y_w': y_w,
                        'y_s': y_s,
                        'y_w_ng': y_w_ng,
                        'y_s_ng': y_s_ng,
                        'y_cls_w': y_cls_w,
                        'y_cls_s': y_cls_s,
                        'y_cls_w_ng': y_cls_w_ng,
                        'y_cls_s_ng': y_cls_s_ng,
                    }, 
                    batch_size_s=batch_size_s, 
                    batch_size_t=batch_size_t
                )
                
                # Generate target pseudo-labels
                max_prob, pseudo_labels_t = torch.max(torch.softmax(y['y_w']['trg'], dim=1), dim=1)
                
                # Empirical risk minimization loss on source domain
                ce_loss = self.ce_loss(y['y_w']['src'], batch['labels_s'])
                
                # Self-training loss with pseudo-labels on target domain
                st_loss = (self.st_loss(y['y_s']['trg'], pseudo_labels_t)
                           * max_prob.ge(self.threshold).float().detach()).mean()
                
                # Entropy loss on target logits
                ts_loss = self.ts_loss(y['y_w']['trg'])
                
                # Cluster loss (for target domain clustering)
                bce_loss, sim_matrix_ulb = self.cluster_loss(
                    f=feat_w,
                    l1=y_cls_w if back_cluster else y_cls_w_ng,
                    l2=y_cls_s if back_cluster else y_cls_s_ng,
                    y=torch.cat((batch['labels_s'], batch['labels_t_scrambled']), dim=0)
                )
                
                # Generate target pseudo-labels for cluster head
                max_prob_cls, pseudo_labels_cls_t = torch.max(
                    torch.softmax(y['y_cls_w']['trg'], dim=1), dim=1
                )
                
                # Self-training loss with pseudo-labels on target domain for cluster head
                st_loss_cls = (self.st_loss(y['y_cls_s']['trg'], pseudo_labels_cls_t)
                               * max_prob_cls.ge(self.threshold).float().detach()).mean()
                
                # Classification head consistent with cluster head
                p_w_ng_t = torch.softmax(y['y_w_ng']['trg'], dim=1) # Target probs from weakly augmented nograd inputs
                p_s_ng_t = torch.softmax(y['y_s_ng']['trg'], dim=1) # Target probs from strongly augmented nograd inputs
                
                pairs1, _ = pair_enum(p_w_ng_t)
                _, pairs2 = pair_enum(p_s_ng_t)

                con_loss_t = self.bce_loss(pairs1, pairs2, sim_matrix_ulb)
                
                # Cluster head consistent with source labels
                # Compute similarity matrix for source labels
                labels_s_view = batch['labels_s'].contiguous().view(-1, 1) # (N_s, 1)
                sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(self.device) # (N_s, N_s)
                sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # 1 for same class, -1 for different class
                
                p_cls_w_ng_s = torch.softmax(y['y_cls_w_ng']['src'], dim=1) # Source cluster probs from weakly augmented nograd inputs
                p_cls_s_ng_s = torch.softmax(y['y_cls_s_ng']['src'], dim=1) # Source cluster probs from strongly augmented nograd inputs
                
                pairs1, _ = pair_enum(p_cls_w_ng_s)
                _, pairs2 = pair_enum(p_cls_s_ng_s)
                
                con_loss_s = self.bce_loss(pairs1, pairs2, sim_matrix_lb.flatten())
                
                # Consistency loss
                con_loss = con_loss_t + con_loss_s
                
                # Invariant loss
                p_w_t = torch.softmax(y['y_w']['trg'], dim=1)
                p_s_t = torch.softmax(y['y_s']['trg'], dim=1)
                
                pairs1, _ = pair_enum(p_w_t)
                _, pairs2 = pair_enum(p_s_t)
                
                inv_loss_t = self.bce_loss(pairs1, pairs2, sim_matrix_ulb)
                
                p_w_s = torch.softmax(y['y_w']['src'], dim=1)
                p_s_s = torch.softmax(y['y_s']['src'], dim=1)
                
                pairs1, _ = pair_enum(p_w_s)
                _, pairs2 = pair_enum(p_s_s)
                
                inv_loss_s = self.bce_loss(pairs1, pairs2, sim_matrix_lb.flatten())
                
                inv_loss = torch.var(torch.stack([inv_loss_t, inv_loss_s]))
                
                # Compute final loss
                loss = self.w_transfer * ts_loss \
                       + bce_loss \
                       + ce_loss \
                       + con_loss \
                       + self.w_self_training * st_loss \
                       + 0.5 * st_loss_cls \
                       + self.w_inv * inv_loss
                       
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                # Metrics
                preds_s = torch.argmax(y['y_w']['src'], dim=1)
                all_labels.append(batch['labels_s'])
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
          
    
    def _prepare_batches(self, src_dataloader, trg_dataloader):
        # Get batches
        (x_s_w, x_s_s), labels_s = next(src_dataloader)
        (x_t_w, x_t_s), labels_t = next(trg_dataloader)
        
        # Pass data to device
        x_s_w = x_s_w.to(self.device)
        x_s_s = x_s_s.to(self.device)
        x_t_w = x_t_w.to(self.device)
        x_t_s = x_t_s.to(self.device)
        labels_s = labels_s.to(self.device).long()
        labels_t = labels_t.to(self.device).long()
        
        # Set fake labels for target domain
        labels_t_scrambled = torch.ones_like(labels_t).to(self.device) + self.num_classes
        
        return {
            'x_s_w': x_s_w, 'x_s_s': x_s_s, 'labels_s': labels_s,
            'x_t_w': x_t_w, 'x_t_s': x_t_s, 'labels_t': labels_t,
            'labels_t_scrambled': labels_t_scrambled
        }  
        
    def _split_logits(self, outputs, batch_size_s, batch_size_t):
        return {
            name: {
                'src': tensor[:batch_size_s],
                'trg': tensor[batch_size_s:batch_size_s + batch_size_t]
            }
            for name, tensor in outputs.items()
        }
                         