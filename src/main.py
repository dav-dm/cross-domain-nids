from util.args_parser import parse_arguments
from util.config import config_threads
from util.seed import seed_everything

from util.logger import Logger
from data.data_manager import DataManager
from trainer.trainer import Trainer


def main():
    # 1. Parse arguments
    args = parse_arguments()
    
    # 2. Set the seed and number of threads
    seed_everything(args.seed)
    config_threads(args.n_thr)
    
    # 3. Init the DataManager
    data_manager = DataManager(args)
    data_manager.prepare_datasets()
    
    # 4. Initialize the Trainer and start the run
    trainer = Trainer(args)
    trainer.run(data_manager)
    
    # 5. Final logging (metrics, graphs, etc.)
    logger = Logger(src_dataset_name=args.src_dataset, trg_dataset_name=args.trg_dataset)
    logger.process_results()
    logger.plot_per_epoch_metrics()


if __name__ == '__main__':
    main()
