import numpy as np
from hmmlearn.hmm import GaussianHMM as SkGaussianHMM
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config

class GHMM(MLModule):
    """
    Wrapper of GaussianHMM from hmmlearn.
    Attributes:
        n_components (int): Number of states in the model. Default is loaded from configuration.
        covariance_type (str): Type of covariance parameters. Default is loaded from configuration.
        n_iter (int): Number of iterations to perform. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.n_components = kwargs.get('ghmm_n_components', cf['ghmm_n_components'])
        self.covariance_type = kwargs.get('ghmm_covariance_type', cf['ghmm_covariance_type'])

        self.classifier = SkGaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--ghmm-n-components', type=int, default=cf['ghmm_n_components'])
        parser.add_argument('--ghmm-covariance-type', type=str, default=cf['ghmm_covariance_type'])
        return parser
    
    def _fit(self, data, _):
        self.classifier.fit(data)

    def _predict(self, data, labels):
        preds = self.classifier.predict(data)
        
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'labels': labels,
            'preds': preds
        }