import numpy as np
from sklearn.semi_supervised import LabelPropagation as SkLabelPropagation
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config


class LabelPropagation(MLModule):
    """
    Wrapper of LabelPropagation from sklearn.
    Attributes:
        kernel (str): The kernel to use. Default is loaded from configuration.
        gamma (float): The gamma parameter for the RBF kernel. Default is loaded from configuration.
        n_neighbors (int): The number of neighbors to use. Default is loaded from configuration.
        max_iter (int): The maximum number of iterations. Default is loaded from configuration.
        tol (float): The tolerance for stopping criteria. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.kernel = kwargs.get('lp_kernel', cf['lp_kernel'])
        self.gamma = kwargs.get('lp_gamma', cf['lp_gamma'])
        self.n_neighbors = kwargs.get('lp_n_neighbors', cf['lp_n_neighbors'])
        self.max_iter = kwargs.get('lp_max_iter', cf['lp_max_iter'])
        self.tol = kwargs.get('lp_tol', cf['lp_tol'])
        
        self.classifier = SkLabelPropagation(
            kernel=self.kernel,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--lp-criterion', type=str, default=cf['lp_kernel'])
        parser.add_argument('--lp-gamma', type=float, default=cf['lp_gamma'])
        parser.add_argument('--lp-n-neighbors', type=int, default=cf['lp_n_neighbors'])
        parser.add_argument('--lp-max-iter', type=int, default=cf['lp_max_iter'])
        parser.add_argument('--lp-tol', type=float, default=cf['lp_tol'])
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