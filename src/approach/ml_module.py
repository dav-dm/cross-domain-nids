import importlib
from argparse import ArgumentParser

from util.config import load_config

ml_approaches = {
    'label_propagation' : 'LabelPropagation',
    'label_spreading' : 'LabelSpreading',
    'st_random_forest' : 'SelfTrainingRandomForest',
    'isolation_forest' : 'IsolationForest',
    'oc_svm' : 'OC_SVM',
    'lof' : 'LocalOutlierFactor',
    'ghmm' : 'GHMM',
    'dae3' : 'DAE3',
}
ml_unsup_approaches = [
    'label_propagation', 'label_spreading', 'st_random_forest'
]
ml_anomaly_detectors = [
    'isolation_forest', 'oc_svm', 'lof', 'ghmm', 'dae3'
]

class MLModule:
    
    def __init__(self, datamodule=None, callbacks=None, **kwargs):
        cf = load_config()

        self.datamodule = datamodule
        self.seed = kwargs.get('seed', cf['seed'])
        self.task = 'src'
        self.phase = None
        
        self.callbacks = callbacks if callbacks is not None else []
        
    @staticmethod
    def get_approach(appr_name, **kwargs):
        Approach = getattr(importlib.import_module(
            name=f'approach.{appr_name}'), ml_approaches[appr_name])
        return Approach(**kwargs)
    
    @staticmethod
    def add_appr_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        return parser
    
    def fit(self):
        self.phase = 'train'
        
        for cb in self.callbacks: 
            cb.on_fit_start(self)
            
        data, labels = self.datamodule.get_train_data()
        self._fit(data, labels)
        
        for cb in self.callbacks:
            cb.on_fit_end(self)
        
    def test(self):
        self.phase = 'test'
        
        for cb in self.callbacks:
            cb.on_test_start(self)
            
        data, labels = self.datamodule.get_test_data()
        self.outputs = self._predict(data, labels)
        
        for cb in self.callbacks:
            cb.on_test_end(self)
        
    def validate(self):
        self.phase = 'val'
        
        for cb in self.callbacks:
            cb.on_validation_start(self)
            
        data, labels = self.datamodule.get_val_data()
        self.outputs = self._predict(data, labels)
        
        for cb in self.callbacks:
            cb.on_validation_end(self)