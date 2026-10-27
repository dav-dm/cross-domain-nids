from pathlib import Path

from callback.callback_lib import Callback
from util.directory_manager import DirectoryManager


class ModelCheckpoint(Callback):
    """
    Callback for checkpointing: saves the model when the monitored metric improves
    and loads the checkpoint at the beginning of validation and test.
    """
    def __init__(self, monitor, mode):
        super().__init__()
        
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.ckpt_path = None
        self.checkpoint_filename = 'checkpoint'  # Base name for the checkpoint file

    def on_epoch_end(self, module, epoch):
        """
        At the end of each epoch, checks if the monitored metric has improved
        compared to the best score obtained so far. If yes, saves the model.
        """
        if self.monitor not in module.epoch_outputs:
            raise ValueError(f'Monitor value "{self.monitor}" not found in module outputs.')

        if self.mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'")
        
        current = module.epoch_outputs[self.monitor]
                
        improved = False
        if self.best_score is None:
            # First epoch -> self.best_score is None
            improved = True
        else:
            if self.mode == 'min' and current < self.best_score:
                improved = True
            elif self.mode == 'max' and current > self.best_score:
                improved = True

        if improved:
            if self.ckpt_path is not None:
                Path(self.ckpt_path).unlink(missing_ok=True)  # Remove old checkpoint_ep.pt
            self.best_score = current
            dm = DirectoryManager()
            path = dm.mkdir('checkpoint') # Create checkpoint directory if it doesn't exist
            checkpoint_path = f'{path}/{self.checkpoint_filename}_{epoch+1}.pt'
            self.ckpt_path = module.save_checkpoint(checkpoint_path)
            
    def _load_checkpoint(self, module, phase):
        if self.ckpt_path is None:
            print(f'[ModelCheckpoint] Checkpoint path is None, using current state.')
            return
        
        if Path(self.ckpt_path).exists():
            print(f"[ModelCheckpoint] Loading checkpoint from {self.ckpt_path} for {phase}.")
            module.load_checkpoint(self.ckpt_path)
        else:
            raise FileNotFoundError(f'Checkpoint file not found at {self.ckpt_path}')

    def on_validation_start(self, module):
        """
        At the start of validation, loads the checkpoint if it exists.
        """
        self._load_checkpoint(module, phase='validation')
        
    def on_test_start(self, module):
        """
        At the start of testing, loads the checkpoint if it exists.
        """
        self._load_checkpoint(module, phase='testing')