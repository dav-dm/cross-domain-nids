import logging

from callback.callback_lib import Callback
from util.directory_manager import DirectoryManager

def _get_logger(name, log_file, formatter, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_file)        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level) 
    return logger


class EarlyStopping(Callback):
    """
    EarlyStopping callback to stop training when a monitored metric has stopped improving.
    Attributes:
        monitor (str): The metric to be monitored.
        mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; 
                    in 'max' mode it will stop when the quantity monitored has stopped increasing.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints a message when training is stopped early.
    """
    
    def __init__(self, monitor, mode, patience, min_delta, verbose=True):
        super().__init__()
        
        self.patience = patience if patience != -1 else float('inf')
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Inner variables
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_adaptation_start(self, module):
        self._init_inner_vars(module)
    
    def on_fit_start(self, module):
        self._init_inner_vars(module)
        
    def _init_inner_vars(self, module):
        # Init inner vars and module.should_stop to False
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        module.should_stop = False
                
    def on_epoch_end(self, module, epoch):
        if self.monitor not in module.epoch_outputs:
            raise ValueError(f'Monitor value "{self.monitor}" not found in module outputs.')
        
        if self.mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'")
        
        current = module.epoch_outputs[self.monitor]
        
        # First epoch, set self.best_score and return
        if self.best_score is None:
            self.best_score = current
            self._log_early_stopping(current, epoch)
            return
        
        improved = False
        
        if self.mode == 'min' and (self.best_score - current > self.min_delta):
            improved = True
        elif self.mode == 'max' and (current - self.best_score > self.min_delta):
            improved = True
                    
        if improved:
            # Model has improved, reset self.wait
            self.best_score = current
            self.wait = 0
        else:
            # Model has not improved
            self.wait += 1
        
            # Check patience and min_epochs
            if self.wait >= self.patience and epoch >= module.min_epochs:
                # Stop training
                module.should_stop = True
                self.stopped_epoch = epoch + 1 
                
        self._log_early_stopping(current, epoch)
                
    def _log_early_stopping(self, current, epoch):
        # Logging
        dm = DirectoryManager()
        logger = _get_logger( 
            f'{dm.log_dir}_es', f'{dm.log_dir}/early_stopping.log',
            logging.Formatter('%(asctime)s - %(message)s')
        )
        logger.info(
            f'epoch{epoch+1}, {self.monitor}_current:{current}, {self.monitor}_best:{self.best_score}, '+
            f'wait_count:{self.wait}, patience:{self.patience}'
        )            
                    
    def on_fit_end(self, _):
        if self.stopped_epoch > 0 and self.verbose:
            print(f'[EarlyStopping] Training stopped at epoch {self.stopped_epoch}.' 
                  f' Best val {self.monitor} was {self.best_score:.6f}.')