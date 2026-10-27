import torch
from torch import nn
from torchmetrics import Accuracy

from module.gradient_reverse_function import WarmStartGradientReverseLayer
    
    
class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dann.py)
    """
    def __init__(self, domain_discriminator, reduction='mean', grl=None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = grl or WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True
        ) 
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction
        self.domain_discriminator_accuracy = None

    def forward(self, f_s, f_t, w_s=None, w_t=None):
        device = f_s.device
        # Concatenate source and target features, then apply gradient reversal
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        # Pass concatenated features through the domain discriminator
        d = self.domain_discriminator(f)
        
        batch_size_s, batch_size_t = f_s.size(0), f_t.size(0)
        
        # Split the discriminator output back into source/target
        d_s, d_t = torch.split(d, [batch_size_s, batch_size_t], dim=0)
        
        # Construct labels: source=1, target=0
        d_label_s = torch.ones(batch_size_s, 1, device=device)
        d_label_t = torch.zeros(batch_size_t, 1, device=device)
        
        # Calculate domain discriminator accuracy for monitoring
        with torch.no_grad():
            bin_accuracy = Accuracy(task='binary').to(device)
            acc_s = bin_accuracy(d_s, d_label_s).item()
            acc_t = bin_accuracy(d_t, d_label_t).item()
            self.domain_discriminator_accuracy = 0.5 * (acc_s + acc_t)

        # Set default weights if None
        w_s = torch.ones_like(d_label_s) if w_s is None else w_s.view_as(d_s)
        w_t = torch.ones_like(d_label_t) if w_t is None else w_t.view_as(d_t)
            
        # Compute BCE losses for source and target, then average
        loss_s = nn.functional.binary_cross_entropy(
            d_s, d_label_s, weight=w_s, reduction=self.reduction)
        loss_t = nn.functional.binary_cross_entropy(
            d_t, d_label_t, weight=w_t, reduction=self.reduction)
        return 0.5 * (loss_s + loss_t)
            
            
class ConditionalDomainAdversarialLoss(nn.Module):
    """The Conditional Domain Adversarial Loss used in 
    `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/cdan.py)
    """
    def __init__(self, domain_discriminator, entropy_conditioning, grl=None, reduction='mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.grl = grl or WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True
        )
        self.domain_discriminator = domain_discriminator
        self.entropy_conditioning = entropy_conditioning
        self.reduction = reduction
        self.domain_discriminator_accuracy = None
    
    def forward(self, f_s, l_s, f_t, l_t):
        device = f_s.device
        embeddings = torch.cat((f_s, f_t), dim=0)
        logits = torch.cat((l_s, l_t), dim=0)
        
        batch_size = embeddings.size(0)
        batch_size_s, batch_size_t = f_s.size(0), f_t.size(0)
        
        preds = torch.softmax(logits, dim=1).detach()
        log_preds = torch.log_softmax(logits, dim=1).detach()
        
        # Apply gradient reversal layer to the linear mapping of the embeddings and predictions
        h = self.grl(self._linear_map(embeddings, preds))
        d = self.domain_discriminator(h)
        
        d_label = torch.cat([
            torch.ones(batch_size_s, 1, device=device),
            torch.zeros(batch_size_t, 1, device=device)
        ])
        with torch.no_grad():
            bin_accuracy = Accuracy(task='binary').to(device)
            self.domain_discriminator_accuracy = bin_accuracy(d, d_label).item()
        
        if self.entropy_conditioning:
            # Entropy = - \sum p(x) log p(x)
            # Compute the entropy weight (1 + e^{-entropy}), then normalize by sum
            entropy = -(preds * log_preds).sum(dim=1)
            entropy_weight = 1.0 + torch.exp(-entropy)
            entropy_weight = entropy_weight / entropy_weight.sum() * batch_size
            weights = entropy_weight.view_as(d)
        else:
            weights = None
        return nn.functional.binary_cross_entropy(
            d, d_label, weight=weights, reduction=self.reduction
        )
        
    def _linear_map(self, f, y):
        batch_size = f.size(0)
        output = torch.bmm(y.unsqueeze(2), f.unsqueeze(1)) # shape: [batch_size, num_classes, feature_dim]
        return output.view(batch_size, -1) # shape: [batch_size, num_classes * feature_dim]
    
            
class MinimumClassConfusionLoss(nn.Module):
    """
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.
    Proposed in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/self_training/mcc.py#L17)
    """
    def __init__(self, T):
        super(MinimumClassConfusionLoss, self).__init__()
        self.T = T

    def forward(self, logits):
        batch_size, num_classes = logits.shape
        
        # Temperature scaling
        preds = torch.softmax(logits / self.T, dim=1)         # shape: [batch_size, num_classes]
        log_preds = torch.log_softmax(logits / self.T, dim=1) # shape: [batch_size, num_classes]
        
        # Entropy = - \sum p(x) log p(x)
        with torch.no_grad():
            entropy = -(preds * log_preds).sum(dim=1) # shape: [batch_size]
            # Compute the entropy weight (1 + e^{-entropy}), then normalize by sum
            entropy_weight = 1.0 + torch.exp(-entropy)
            entropy_weight = (batch_size * entropy_weight / entropy_weight.sum()).unsqueeze(1)
            # shape of entropy_weight: [batch_size, 1]
        
        # Compute the class confusion matrix = P^T * P, weighed by the entropy weights
        # where P = softmax(logits)
        class_confusion_matrix = (preds * entropy_weight).transpose(0, 1).mm(preds)
        # Normalize each row
        class_confusion_matrix = class_confusion_matrix / class_confusion_matrix.sum(dim=1, keepdim=True)
        
        # MCC loss = (sum of matrix - trace of matrix) / num_classes
        # trace is the sum of diagonal elements
        mcc_loss = (class_confusion_matrix.sum() - class_confusion_matrix.trace()) / num_classes
        return mcc_loss     
    
    
class PairwiseBCE(nn.Module):
    """
    Binary Cross-Entropy Loss for similarity learning with pairwise probabilities.
    
    Computes a variant of binary cross-entropy loss for pairs of probability distributions,
    where similarity labels indicate whether distributions should be similar (1),
    dissimilar (-1), or ignored (0).
    """
    def forward(self, prob1, prob2, simi):
        EPS = 1e-7  # Avoid calculating log(0). Use small value of float16.
        # Validate input dimensions
        assert len(prob1) == len(prob2) == len(simi), \
            f'Wrong input size: {len(prob1)}, {len(prob2)}, {len(simi)}'
        
        # Compute element-wise product and sum across classes
        P = prob1.mul(prob2).sum(dim=1)  # shape: [batch_size]
        
        # Transform probabilities based on similarity labels
        # For similar pairs: keep P (we want P to be close to 1)
        # For dissimilar pairs: transform to 1-P (we want P to be close to 0, so 1-P close to 1)
        # For ignored pairs: set to 1 (log(1) = 0, no contribution to loss)
        P_transformed = P.mul(simi).add(simi.eq(-1).type_as(P))
        
        # Compute negative log probability with numerical stability
        neglogP = -torch.log(P_transformed + EPS)
        
        # Mask out ignored pairs (simi == 0)
        mask = simi.ne(0).type_as(neglogP)
        neglogP_masked = neglogP.mul(mask)
        
        # Compute mean over non-ignored pairs
        if mask.sum() > 0:
            return neglogP_masked.sum() / mask.sum()
        else:
            return neglogP_masked.mean()  # fallback if all pairs are ignored
    
    
class ClusterLoss(nn.Module):
    """
    Cluster Loss for semi-supervised learning with pairwise similarity constraints.
    
    Computes pairwise similarity constraints between unlabeled samples using either:
    - Cosine similarity thresholding, or
    - Top-k rank statistics
    
    The loss encourages consistency between predictions from two classifiers
    for pairs deemed similar based on feature-space relationships.
    """
    def __init__(self, num_classes, bce_type, cosine_threshold, topk):
        super().__init__()
        self.num_classes = num_classes
        self.bce_type = bce_type
        self.cos_thre = cosine_threshold
        self.topk = topk
        self.bce = PairwiseBCE()
                
    def forward(self, f, l1, l2, y):
        device = f.device
        
        # Create mask for labeled samples (y < num_classes)
        mask_l = y < self.num_classes  # True for labeled, False for unlabeled 
        
        p1, p2 = torch.softmax(l1, dim=1), torch.softmax(l2, dim=1)
        
        # Extract features of unlabeled samples only
        f_ulb = f[~mask_l].detach()  # Detach to prevent gradient flow through features
        
        if self.bce_type == 'cosine':
            
            f_norm = nn.functional.normalize(f_ulb, dim=1)
            # Enumerate all unordered pairs of unlabeled samples
            f_row, f_col = pair_enum(f_norm)
            # Compute cosine similarity between all pairs
            # Shape: [n_pairs] where n_pairs = n_ulb * (n_ulb - 1) / 2
            cos_sim = torch.bmm(
                f_row.view(f_row.size(0), 1, -1),
                f_col.view(f_col.size(0), -1, 1)
            ).squeeze()
            # Assign similarity targets based on cosine threshold
            # 1 for similar (cos_sim > threshold), -1 for dissimilar otherwise
            target_ulb = torch.zeros_like(cos_sim).float() - 1
            target_ulb[cos_sim > self.cos_thre] = 1
            
        elif self.bce_type == 'RK':
            
            # Top-k rank statistics
            rank_idx = torch.argsort(f_ulb, dim=1, descending=True)
            # Enumerate all unordered pairs of rank indices
            rank_idx1, rank_idx2 = pair_enum(rank_idx)
            # Take only top-k features for comparison
            rank_idx1 = rank_idx1[:, :self.topk]
            rank_idx2 = rank_idx2[:, :self.topk]
            # Sort indices to make comparison order-invariant
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            # Compute rank difference (L1 distance between rank sets)
            rank_diff = torch.abs(rank_idx1 - rank_idx2).sum(dim=1)
            # Assign similarity targets: 1 if identical top-k sets, -1 otherwise
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1
            
        else:
            raise ValueError(f"Unknown bce_type: {self.bce_type}. "
                             f"Expected 'cosine' or 'RK'.")
                   
        p1_ulb = p1[~mask_l]
        p2_ulb = p2[~mask_l]
        
        # Enumerate all unordered probability pairs
        p1_pairs, _ = pair_enum(p1_ulb)
        _, p2_pairs = pair_enum(p2_ulb)
        
        # Compute BCE loss between probability pairs using similarity targets
        bce_loss = self.bce(p1_pairs, p2_pairs, target_ulb)
        return bce_loss, target_ulb
   
    
def pair_enum(x):
    """
    Helper function for ICON and ClusterLoss.
    Enumerate all unordered pairs from a batch.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    x1 = x.repeat(x.size(0), 1) # [batch_size^2, feature_dim]
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1)) # [batch_size^2, feature_dim]
    return x1, x2

    
class TsallisEntropy(nn.Module):
    """
    Tsallis Entropy Loss for self-training and domain adaptation.
    """
    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, logits):
        batch_size, _ = logits.shape
        pred = torch.softmax(logits / self.temperature, dim=1)  # shape: [batch_size, num_classes]
        entropy_vals = self.entropy(pred).detach()
        
        # Compute entropy-based weights: w = 1 + exp(-entropy)
        # Higher entropy (more uncertain) → lower weight, Lower entropy (confident) → higher weight
        entropy_weight = 1.0 + torch.exp(-entropy_vals)
        
        # Normalize weights to sum to batch_size
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        
        # Compute weighted sum of predictions per class: Σ_i w_i * p_i for each class
        weighted_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)  # shape: [1, num_classes]
        
        # Compute Tsallis entropy loss
        # L = 1/(α-1) * Σ_i (1/Σ_j w_j - Σ_i p_i^α / Σ_j w_j * w_i)
        return 1.0 / (self.alpha - 1.0) * torch.sum(
            (1 / torch.mean(weighted_sum) - torch.sum(pred ** self.alpha / weighted_sum * entropy_weight, dim=-1))
        )
            
    @staticmethod
    def entropy(predictions, reduction='none'):
        epsilon = 1e-5  # Small constant to avoid log(0)
    
        # Compute entropy: H(p) = -Σ p_i * log(p_i)
        H = -predictions * torch.log(predictions + epsilon)
        H = H.sum(dim=1)  # Sum over classes
        
        if reduction == 'mean':
            return H.mean()
        else:
            return H
        
     
class ClassifierDiscrepancy(nn.Module):
    """
    Classifier Discrepancy Loss used in MCD
    """
    def __init__(self):
        super().__init__()

    def forward(self, y1, y2):
        p1 = nn.functional.softmax(y1, dim=1)
        p2 = nn.functional.softmax(y2, dim=1)
        return torch.mean(torch.abs(p1 - p2))


class MCDEntropy(nn.Module):
    """
    MCDEntropy Loss used in MCD
    """
    def __init__(self):
        super().__init__()

    def forward(self, y):
        p = nn.functional.softmax(y, dim=1)
        p_mean = torch.mean(p, dim=0)
        return -torch.mean(torch.log(p_mean + 1e-6))
    

class BatchSpectralPenalizationLoss(nn.Module):
    """
    Batch Spectral Penalization (BSP)
    Penalize top-k singular values of the feature matrix of a batch
    """
    def __init__(self):
        super().__init__()

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.linalg.svd(f_s, full_matrices=False)
        _, s_t, _ = torch.linalg.svd(f_t, full_matrices=False)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss