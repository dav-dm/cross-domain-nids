import numpy as np
from sklearn.neighbors import LocalOutlierFactor as SkLocalOutlierFactor
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config

class LocalOutlierFactor(MLModule):
    """
    Wrapper of LocalOutlierFactor from sklearn.
    Attributes:
        n_neighbors (int): Number of neighbors to use by default for kneighbors queries. Default is loaded from configuration.
        contamination (float): The proportion of outliers in the data set. Default is loaded from configuration.
        novelty (bool): Whether to enable novelty detection. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.n_neighbors = kwargs.get('lof_n_neighbors', cf['lof_n_neighbors'])
        self.contamination = kwargs.get('lof_contamination', cf['lof_contamination'])
        self.novelty = kwargs.get('lof_novelty', cf['lof_novelty'])

        self.classifier = SkLocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--lof-n-neighbors', type=int, default=cf['lof_n_neighbors'])
        parser.add_argument('--lof-contamination', type=str, default=cf['lof_contamination'])
        parser.add_argument('--lof-novelty', type=bool, default=cf['lof_novelty'])
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
            'preds': preds
        }