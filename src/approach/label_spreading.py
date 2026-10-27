import numpy as np
from sklearn.semi_supervised import LabelSpreading as SkLabelSpreading
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config


class LabelSpreading(MLModule):
    """
    Wrapper of LabelSpreading from sklearn.
    Attributes:
        kernel (str): The kernel to use. Default is loaded from configuration.
        gamma (float): The gamma parameter for the RBF kernel. Default is loaded from configuration.
        n_neighbors (int): The number of neighbors to use. Default is loaded from configuration.
        alpha (float): The alpha parameter for the label spreading algorithm. Default is loaded from configuration.
        max_iter (int): The maximum number of iterations. Default is loaded from configuration.
        tol (float): The tolerance for stopping criteria. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.kernel = kwargs.get('ls_kernel', cf['ls_kernel'])
        self.gamma = kwargs.get('ls_gamma', cf['ls_gamma'])
        self.n_neighbors = kwargs.get('ls_n_neighbors', cf['ls_n_neighbors'])
        self.alpha = kwargs.get('ls_alpha', cf['ls_alpha'])
        self.max_iter = kwargs.get('ls_max_iter', cf['ls_max_iter'])
        self.tol = kwargs.get('ls_tol', cf['ls_tol'])
        
        self.classifier = SkLabelSpreading(
            kernel=self.kernel,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--ls-kernel', type=str, default=cf['ls_kernel'])
        parser.add_argument('--ls-gamma', type=float, default=cf['ls_gamma'])
        parser.add_argument('--ls-n-neighbors', type=int, default=cf['ls_n_neighbors'])
        parser.add_argument('--ls-alpha', type=float, default=cf['ls_alpha'])
        parser.add_argument('--ls-max-iter', type=int, default=cf['ls_max_iter'])
        parser.add_argument('--ls-tol', type=float, default=cf['ls_tol'])
        return parser
    
    def _fit(self, data, labels):
        self.classifier.fit(data, labels)

    def _predict(self, data, labels):
        probs = self.classifier.predict_proba(data)
        preds = np.argmax(probs, axis=1)
        
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'labels': labels,
            'preds': preds,
        }