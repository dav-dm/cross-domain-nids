import numpy as np
from sklearn.ensemble import IsolationForest as SkIsolationForest 
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config

class IsolationForest(MLModule):
    """
    Wrapper of IsolationForest from sklearn.
    Attributes:
        n_estimators (int): The number of base estimators in the ensemble. Default is loaded from configuration.
        max_samples (int or float): The number of samples to draw to train each base estimator. Default is loaded from configuration.
        contamination (float): The proportion of outliers in the data set. Default is loaded from configuration.
        max_features (int or float): The number of features to draw from X to train each base estimator. Default is loaded from configuration.
        bootstrap (bool): Whether samples are drawn with replacement. Default is loaded from configuration.
        n_jobs (int): The number of jobs to run in parallel. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.n_estimators = kwargs.get('if_n_estimators', cf['if_n_estimators'])
        self.max_samples = kwargs.get('if_max_samples', cf['if_max_samples'])
        self.contamination = kwargs.get('if_contamination', cf['if_contamination'])
        self.max_features = kwargs.get('if_max_features', cf['if_max_features'])
        self.bootstrap = kwargs.get('if_bootstrap', cf['if_bootstrap'])

        
        self.classifier = SkIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--if-n-estimators', type=int, default=cf['if_n_estimators'])
        parser.add_argument('--if-max-samples', type=str, default=cf['if_max_samples'])
        parser.add_argument('--if-contamination', type=str, default=cf['if_contamination'])
        parser.add_argument('--if-max-features', type=float, default=cf['if_max_features'])
        parser.add_argument('--if-bootstrap', type=bool, default=cf['if_bootstrap'])
        return parser
    
    def _fit(self, data, _):
        self.classifier.fit(data)

    def _predict(self, data, labels):
        preds = self.classifier.predict(data)
        # Convert predictions from {-1, 1} to {1, 0}
        preds = np.where(preds == -1, 1, 0)
        
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'labels': labels,
            'preds': preds,
        }