import sys
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from approach.dl_module import DLModule
from module.head import NNHead
from module.loss import DistillKLLoss
from module.teacher import RFSTeacher
from module.ad_domain_discriminator import AutoencoderDetector
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class RFS(DLModule):
    """
    [[Link to Source Code]](https://github.com/RL-VIG/LibFewShot)
    RFS is a class that implements the RFS (Rethinking Few-Shot) approach for few-shot learning,
    as described in "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.kd_T = kwargs.get('kd_t', cf['kd_t'])
        self.is_distill = kwargs.get('is_distill', cf['is_distill'])
        self.teacher_path = kwargs.get('teacher_path', cf['teacher_path'])
        self.discrimator_path = kwargs.get('discr_path', cf['discr_path'])
        self.alpha = kwargs.get('alpha', cf['alpha']) if self.is_distill else 0
        self.gamma = kwargs.get('gamma', cf['gamma']) if self.is_distill else 1
        if self.alpha + self.gamma != 1.0:
            raise ValueError('alpha + gamma should be equal to 1')

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = DistillKLLoss(T=self.kd_T)
        
        self.teacher = RFSTeacher(
            net=self.net, 
            is_distill=self.is_distill, 
            teacher_path=self.teacher_path
        )
        self.domain_discriminator = None
        self.nn_head = None

        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--alpha', type=float, default=cf['alpha'])
        parser.add_argument('--gamma', type=float, default=cf['gamma'])
        parser.add_argument('--is-distill', action='store_true', default=cf['is_distill'])
        parser.add_argument('--kd-t', type=float, default=cf['kd_t'])
        parser.add_argument('--teacher-path', type=str, default=cf['teacher_path'])
        parser.add_argument('--discr-path', type=str, default=cf['discr_path'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        student_logits = self.net(batch_x)
        teacher_logits = self.teacher(batch_x)
        
        # CE on actual label and student logits
        gamma_loss = self.ce_loss(student_logits, batch_y)
        # KL on teacher and student logits
        alpha_loss = self.kl_loss(student_logits, teacher_logits)
        
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha
        return loss, student_logits
                
            
    def _predict_step(self, batch_x, batch_y):
        if self.domain_discriminator is not None and self.phase == 'test':
            # Use the domain discriminator to predict the domain 
            flat_batch_x = np.array([np.ravel(x.T) for x in batch_x.detach().cpu().numpy()])
            domain_pred = torch.as_tensor(
                self.domain_discriminator.predict(flat_batch_x),
                device=batch_x.device,
            )
            src_mask = domain_pred == 1   # inliers  → source domain
            trg_mask = domain_pred == -1  # outliers → target domain
            
            logits = batch_x.new_zeros(batch_x.size(0), self.num_classes)
            
            if src_mask.any():
                logits[src_mask] = self.net(batch_x[src_mask])
                
            if trg_mask.any():
                _, trg_emb = self.net(batch_x[trg_mask], return_feat=True)
                logits[trg_mask] = self.nn_head(trg_emb)    
        else:
            # No domain discriminator, domain oracle mode
            if self.task == 'src':
                logits = self.net(batch_x)
            else:
                _, batch_emb = self.net(batch_x, return_feat=True)
                logits = self.nn_head(batch_emb)

        loss = self.ce_loss(logits, batch_y)
        return loss, logits
        
        
    @torch.no_grad()
    def _adapt(self, adapt_dataloader, val_dataloader, **kwargs):
        # RFS freezes the backbone during the adaptation phase  
        self.net.freeze_backbone()  
        self.net.trainability_info()
        self.net.eval()
        
        embeddings, labels = [], []
        
        adapt_loop = tqdm(
            adapt_dataloader, desc='[fitting NN head]', leave=True, disable=disable_tqdm
        )
        for batch_x, batch_y in adapt_loop:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Embed the input
            _, batch_emb = self.net(batch_x, return_feat=True)
            embeddings.append(batch_emb)
            labels.append(batch_y)
            
        # Add a new nearest neighbor head 
        # self.nn_head = KNNHead()
        self.nn_head = NNHead()
        # Store train embedding and labels
        self.nn_head.fit(x=torch.cat(embeddings), y=torch.cat(labels))
        
        # Load the domain discriminator if path is provided
        if self.discrimator_path is not None:
            ckpt = torch.load(self.discrimator_path, weights_only=True)
            self.domain_discriminator = AutoencoderDetector(
                input_dim = ckpt['input_dim'],
                latent_dim = ckpt['latent_dim'],
                quantile = ckpt['quantile'],
                device = self.device,
            )
            self.domain_discriminator._model.load_state_dict(ckpt['state_dict'])
            self.domain_discriminator.threshold_ = ckpt['threshold']
                