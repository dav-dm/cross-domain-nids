import json

from approach.approach_factory import get_approach
from util.directory_manager import DirectoryManager


class Trainer:
    """
    The Trainer class is responsible for managing the training and evaluation process.
    It initializes with the given arguments, and provides methods to run
    the training, validation, and testing phases using the specified approach.
    """
    def __init__(self, args):
        self.args = args
        self.dict_args = vars(args)
        self.dm = DirectoryManager()
        
        
    def run(self, data_manager):
        self.dict_args['num_classes'] = data_manager.num_classes
                        
        self._save_dict_args()

        # Init approach
        approach = get_approach(self.args.approach, **self.dict_args)
        print('='*100)
        
        # Handle different training scenarios
        if self.args.trg_dataset is None:
            self._handle_sup_src(approach, data_manager)
        elif self.args.is_appr_unsup:
            self._handle_sup_src_unsup_trg(approach, data_manager)
        elif self.args.is_anomaly_dect:
            self._handle_anomaly_detection(approach, data_manager)
        else:
            self._handle_sup_src_sup_trg(approach, data_manager)
            
    
    def _handle_sup_src(self, approach, data_manager):
        # Train, validation, and test only on src_dataset
        approach.datamodule = data_manager.get_datamodule(split='src')
        self._fit_val(approach)
        self._test(approach)
        
        
    def _handle_sup_src_unsup_trg(self, approach, data_manager):
        """
        Handle unsupervised domain adaptation scenario.
        """
        if self.args.n_tasks == 1:
            # n_task == 1 -> monolithic training (only for ML semi-sup appoaches)
            # Combine src_dataset and trg_dataset train/val splits (trg train is unsupervised)
            approach.datamodule = data_manager.concat_src_trg(unsup_trg=True)
            self._fit_val(approach)
            self._cross_dataset_test(approach, data_manager.src_splits, data_manager.trg_splits)
        else:
            # n_task == 2 -> sequential training
            src_datamodule = data_manager.get_datamodule(split='src')
            if not self.args.skip_t1:
                # Train and val on src
                approach.datamodule = src_datamodule # Supervised data from src
                print(f'[Trainer] Starting training on source dataset: {self.args.src_dataset}')
                self._fit_val(approach)
                
            checkpoint_path = self.args.ckpt_path or approach.checkpoint_path

            self.dm.change_log_dir(task='trg') # Switch log_dir to trg 
            
            # Target adaptation
            approach = self._reset_approach(checkpoint_path)
            approach.task = 'trg'
            approach.datamodule = data_manager.get_datamodule(split='trg') # Unsupervised data from trg
            print(f'[Trainer] Starting training on target dataset: {self.args.trg_dataset}')
            self._adapt_val(approach, train_dataloader=src_datamodule.get_adapt_data())
            self._cross_dataset_test(approach, data_manager.src_splits, data_manager.trg_splits) 
            
            
    def _handle_anomaly_detection(self, approach, data_manager):        
        """
        Anomaly detection scenario:
        - Train on benign-only samples from src
        """
        approach.datamodule = data_manager.get_anomaly_datamodule()
        self._fit_val(approach)
        self._cross_dataset_test(approach, data_manager.src_splits, data_manager.trg_splits)
            
            
    def _handle_sup_src_sup_trg(self, approach, data_manager):
        """
        Handle monolithic and sequential training on source and target.
        """
        if self.args.n_tasks == 1:
            # n_task == 1 -> monolithic training
            # Combine src_dataset and trg_dataset train/val splits
            approach.datamodule = data_manager.concat_src_trg()
            self._fit_val(approach)
            self._cross_dataset_test(approach, data_manager.src_splits, data_manager.trg_splits)
        else:
            # n_task == 2 -> sequential training
            if not self.args.skip_t1:
                # Train and val on src
                approach.datamodule = data_manager.get_datamodule(split='src')
                print(f'[Trainer] Starting training on source dataset: {self.args.src_dataset}')
                self._fit_val(approach)
            
            checkpoint_path = self.args.ckpt_path or approach.checkpoint_path
            
            self.dm.change_log_dir(task='trg') # Switch the log_dir to trg 
            
            # Target adaptation
            if not self.args.skip_t2:
                approach = self._reset_approach(checkpoint_path)
                approach.task = 'trg'
                approach.datamodule = data_manager.get_datamodule(split='trg')
                print(f'[Trainer] Starting training on target dataset: {self.args.trg_dataset}')
                self._adapt_val(approach)
            self._cross_dataset_test(approach, data_manager.src_splits, data_manager.trg_splits)
        
        
    def _cross_dataset_test(self, approach, src_splits, trg_splits):
        # Test on target
        approach.task = 'trg'
        print(f'[Trainer] Starting test on target dataset: {self.args.trg_dataset}')
        self.dm.change_log_dir(task='trg')
        self._test(approach, test_dataset=trg_splits.test)

        # Test on source
        print(f'[Trainer] Starting test on source dataset: {self.args.src_dataset}')
        approach.task = 'src'
        self.dm.change_log_dir(task='src')
        self._test(approach, test_dataset=src_splits.test)
         
    def _fit_val(self, approach):
        approach.fit()
        approach.validate()
        
    def _adapt_val(self, approach, **kwargs):
        approach.adapt(**kwargs)
        approach.validate()
        
    def _test(self, approach, test_dataset=None):
        if test_dataset is not None:
            approach.datamodule.set_test_dataset(test_dataset)     
        approach.test()
     
    def _reset_approach(self, checkpoint_path):        
        approach = get_approach(approach_name=self.args.approach, **self.dict_args)
        # Loads checkpoint from the first task or from checkpoint_path
        approach.load_checkpoint(checkpoint_path) 
        print(f'[Trainer] Loaded approach state from {checkpoint_path}')
        return approach  
            
    def _save_dict_args(self):
        with open(f'{self.dm.log_dir}/dict_args.json', 'w') as f:
            json.dump(self.dict_args, f)
    