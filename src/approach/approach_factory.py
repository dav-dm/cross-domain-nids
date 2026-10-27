from approach.ml_module import MLModule, ml_approaches, ml_unsup_approaches, ml_anomaly_detectors
from approach.dl_module import DLModule, dl_approaches, dl_unsup_approaches
from util.config import load_config
from callback import (
    EarlyStopping,
    SaveOutputs,
    ModelCheckpoint,
    SaveTrainLog,
    TimeMeasurement,
)


def get_approach_type(approach_name):
    if approach_name in ml_approaches:
        return 'ml'
    elif approach_name in dl_approaches:
        return 'dl'
    else:
        raise ValueError(f"Approach '{approach_name}' not found in ML or DL approaches.")
    
def is_approach_unsupervised(approach_name):
    return approach_name in dl_unsup_approaches + ml_unsup_approaches

def approach_requires_transformations(approach_name):
    return approach_name in ['icon']

def is_anomaly_detector(approach_name):
    return approach_name in ml_anomaly_detectors

def get_approach(approach_name, datamodule=None, **kwargs):
    callbacks = [SaveOutputs(), TimeMeasurement()] # Base callbacks for both ML and DL approaches
    cf = load_config()
    appr_type = kwargs.get('appr_type', None)
    
    if appr_type == 'ml':
        return MLModule.get_approach(
            appr_name=approach_name,
            datamodule=datamodule,
            callbacks=callbacks,
            **kwargs
        )
    elif appr_type == 'dl':
        callbacks.extend([
            EarlyStopping(
                monitor=cf['es_monitor'],
                mode=cf['es_mode'],
                patience=cf['es_patience'],
                min_delta=cf['es_min_delta']
            ),
            ModelCheckpoint(
                monitor=cf['mc_monitor'],
                mode=cf['mc_mode']
            ),
            SaveTrainLog()
        ])
        return DLModule.get_approach(
            appr_name=approach_name,
            datamodule=datamodule,
            callbacks=callbacks,
            **kwargs
        )   
    else:
        raise ValueError(f"Approach '{approach_name}' not found in ML or DL approaches.")