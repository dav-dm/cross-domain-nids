import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config

class SelfTrainingRandomForest(MLModule):
    """
    Wrapper of SelfTrainingClassifier with RandomForestClassifier as base estimator.
    Attributes:
        TODO
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.criterion = kwargs.get('st_criterion', cf['st_criterion'])
        self.threshold = kwargs.get('st_threshold', cf['st_threshold'])
        self.k_best = kwargs.get('st_k_best', cf['st_k_best'])
        self.max_iters = kwargs.get('st_max_iters', cf['st_max_iters'])
        
        # Random Forest estimator args
        rf_criterion = kwargs.get('rf_criterion', cf['rf_criterion'])
        rf_n_estimators = kwargs.get('rf_n_estimators', cf['rf_n_estimators'])
        rf_max_depth = kwargs.get('rf_max_depth', cf['rf_max_depth'])
        rf_estimator = RandomForestClassifier(
            random_state=self.seed,
            criterion=rf_criterion,
            n_estimators=rf_n_estimators, 
            max_depth=rf_max_depth,
        )
        
        self.classifier = SelfTrainingClassifier(
            estimator=rf_estimator,
            criterion=self.criterion,
            threshold=self.threshold,
            k_best=self.k_best,
            max_iter=self.max_iters,
            verbose=True
        )
                    
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--st-criterion', type=str, default=cf['st_criterion'])
        parser.add_argument('--st-threshold', type=float, default=cf['st_threshold'])
        parser.add_argument('--st-k-best', type=int, default=cf['st_k_best'])
        parser.add_argument('--st-max-iters', type=int, default=cf['st_max_iters'])
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
