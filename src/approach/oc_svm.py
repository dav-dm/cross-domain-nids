import numpy as np
from sklearn.svm import OneClassSVM as SkOneClassSVM
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config

class OC_SVM(MLModule):
    """
    Wrapper of OneClassSVM from sklearn.
    Attributes:
        kernel (str): Specifies the kernel type to be used in the algorithm. Default is loaded from configuration.
        nu (float): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Default is loaded from configuration.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.kernel = kwargs.get('ocsvm_kernel', cf['ocsvm_kernel'])
        self.degree = kwargs.get('ocsvm_degree', cf['ocsvm_degree'])
        self.nu = kwargs.get('ocsvm_nu', cf['ocsvm_nu'])
        self.gamma = kwargs.get('ocsvm_gamma', cf['ocsvm_gamma'])
        
        self.classifier = SkOneClassSVM(
            kernel=self.kernel,
            degree=self.degree,
            nu=self.nu,
            gamma=self.gamma,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--ocsvm-kernel', type=str, default=cf['ocsvm_kernel'])
        parser.add_argument('--ocsvm-degree',type=int, default=cf['ocsvm_degree'])
        parser.add_argument('--ocsvm-nu', type=float, default=cf['ocsvm_nu'])
        parser.add_argument('--ocsvm-gamma', type=str, default=cf['ocsvm_gamma'])
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