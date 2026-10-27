import json
import numpy as np

from util.directory_manager import DirectoryManager
from callback.callback_lib import Callback
 

class SaveOutputs(Callback):
    """
    Save prediction results and metrics 
    """
    def _save_outputs(self, module):
        dm = DirectoryManager()
        path = dm.mkdir(f'{module.phase}')
            
        if 'labels' in module.outputs:
            np.savez_compressed(f'{path}/labels.npz', labels=module.outputs['labels'])
        
        if 'preds' in module.outputs:
            np.savez_compressed(f'{path}/preds.npz', preds=module.outputs['preds'])
        
        if 'logits' in module.outputs:
            np.savez_compressed(f'{path}/logits.npz', logits=module.outputs['logits'])
        
        metrics_to_save = {}
        for key in ['accuracy', 'f1_score_macro', 'loss']:
            if key in module.outputs:
                metrics_to_save[key] = module.outputs[key]
        
        with open(f'{dm.log_dir}/{module.phase}_results.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=4)

    def on_test_end(self, module):
        self._save_outputs(module)

    def on_validation_end(self, module):
        self._save_outputs(module)