import pandas as pd
from pathlib import Path

from callback.callback_lib import Callback
from util.directory_manager import DirectoryManager


class SaveTrainLog(Callback):
    def __init__(self, filename='epoch_metrics', write_interval=10):
        """
        Callback that, at the end of each epoch, collects the training and validation
        postfix metrics and appends them as a new row to a file on disk.
        
        The file is stored in the "epoch_metrics" directory with the name "epoch_metrics.csv".
        """
        self.filename = filename
        self.write_interval = write_interval
        # Store rows in memory before writing
        self.metrics_buffer = []

    def on_epoch_end(self, module, epoch):
        metrics = {'epoch': epoch + 1}
        
        if module.epoch_outputs is not None:
            metrics['trn_loss'] = module.epoch_outputs.get('train_loss', 0)
            metrics['trn_acc']  = module.epoch_outputs.get('train_accuracy', 0)
            metrics['trn_f1']   = module.epoch_outputs.get('train_f1_score_macro', 0)
            metrics['val_acc']  = module.epoch_outputs.get('accuracy', 0)
            metrics['val_f1']   = module.epoch_outputs.get('f1_score_macro', 0)
            metrics['val_loss'] = module.epoch_outputs.get('loss', 0)
        else:
            raise ValueError('module.epoch_outputs is None, cannot collect epoch metrics.')

        self.metrics_buffer.append(metrics)

        if (epoch + 1) % self.write_interval == 0:
            self._flush_to_disk()

    def on_adaptation_end(self, _):
        # Optionally flush anything left at the end
        if len(self.metrics_buffer) > 0:
            self._flush_to_disk()
            
    def on_fit_end(self, _):
        # Optionally flush anything left at the end
        if len(self.metrics_buffer) > 0:
            self._flush_to_disk()
    
    def _flush_to_disk(self):
        fullpath = Path(DirectoryManager().log_dir) / f'{self.filename}.csv'
        df_new_data = pd.DataFrame(self.metrics_buffer)
        if fullpath.exists():
            existing_df = pd.read_csv(fullpath)
            combined = pd.concat([existing_df, df_new_data], ignore_index=True)
            combined.to_csv(fullpath, index=False)
        else:
            df_new_data.to_csv(fullpath, index=False)
        self.metrics_buffer = []
