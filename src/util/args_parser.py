from argparse import ArgumentParser

from data.data_module import DataModule
from util.config import load_config
from util.directory_manager import DirectoryManager
from module.gradient_reverse_function import WarmStartGradientReverseLayer
from approach import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingRandomForest,
    IsolationForest,
    OC_SVM,
    LocalOutlierFactor,
    GHMM,
    DAE3,
    Baseline,
    ADDA,
    MCC,
    MCD,
    BSP,
    CDAN,
    ICON,
    get_approach_type,
    is_approach_unsupervised,
    is_anomaly_detector,
    approach_requires_transformations,
    get_approach_type,
)


def parse_arguments():
    cf = load_config()
    
    # Experiment args
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = LabelPropagation.add_appr_specific_args(parser)
    parser = LabelSpreading.add_appr_specific_args(parser)
    parser = SelfTrainingRandomForest.add_appr_specific_args(parser)
    parser = IsolationForest.add_appr_specific_args(parser)
    parser = OC_SVM.add_appr_specific_args(parser)
    parser = LocalOutlierFactor.add_appr_specific_args(parser)
    parser = GHMM.add_appr_specific_args(parser)
    parser = DAE3.add_appr_specific_args(parser)
    parser = Baseline.add_appr_specific_args(parser)
    parser = ADDA.add_appr_specific_args(parser)
    parser = MCC.add_appr_specific_args(parser)
    parser = MCD.add_appr_specific_args(parser)
    parser = CDAN.add_appr_specific_args(parser)
    parser = BSP.add_appr_specific_args(parser)
    parser = ICON.add_appr_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = WarmStartGradientReverseLayer.add_specific_args(parser)
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--gpu', action='store_true', default=cf['gpu'], help='Use GPU if available')
    parser.add_argument('--n-thr', type=int, default=cf['n_thr'], help='Number of threads')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory')
    parser.add_argument('--n-tasks', type=int, default=cf['n_task'], choices=[1, 2], 
                        help='with 1 the model is trained on both src and trg dataset at the same time,\
                              with 2 the model is first trained on src then on trg')
    parser.add_argument('--approach', type=str, default=cf['approach'], help='ML or DL approach to use')
    parser.add_argument('--network', type=str, default=cf['network'], help='Network to use')
    parser.add_argument('--ckpt-path', type=str, default=cf['ckpt_path'], 
                        help='Path to the .pt file containing the state of an approach')
    parser.add_argument('--skip-t1', action='store_true', default=cf['skip_t1'], 
                        help='Skip the first task on src dataset, used only when n_task 2')
    parser.add_argument('--skip-t2', action='store_true', default=cf['skip_t2'], 
                        help='Skip the second task on trg dataset, used only when n_task 2')
    # Data args
    parser.add_argument('--src-dataset', type=str, default=cf['src_dataset'], 
                        help='Source dataset to use')
    parser.add_argument('--trg-dataset', type=str, default=cf['trg_dataset'], 
                        help='Target dataset to use')
    parser.add_argument('--is-flat', action='store_true', default=cf['is_flat'],
                        help='Flat the PSQ input')
    parser.add_argument('--num-pkts', type=int, default=cf['num_pkts'], 
                        help='Number of packets to consider in each biflow')
    parser.add_argument('--fields', type=str, default=cf['fields'],  
                        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
                        help='Field or fields used (default=%(default)s)', 
                        nargs='+', metavar='FIELD')
    
    args = parser.parse_args()
    
    args.appr_type = get_approach_type(args.approach) 
    args.is_appr_unsup = is_approach_unsupervised(args.approach)
    args.is_anomaly_dect = is_anomaly_detector(args.approach)
    args.apply_transformations = approach_requires_transformations(args.approach)
    
    if args.src_dataset is None:
        raise ValueError(f'Source Dataset is None')
    
    if args.trg_dataset==args.src_dataset:
        raise ValueError(f"Target dataset cannot be the same as source dataset: '{args.src_dataset}")
    
    if args.appr_type == 'ml' and args.n_tasks > 1:
        raise ValueError('ML approaches do not support multiple tasks')
    
    if args.is_appr_unsup and args.appr_type == 'dl' and args.n_tasks != 2:
        raise ValueError('Unsupervised DL approaches only support 2 tasks')
    
    if args.skip_t1 and args.n_tasks==2:
        print('WARNING: skipping task on src dataset')
        
    if args.skip_t2 and args.n_tasks==2:
        print('WARNING: skipping task on trg dataset')
        
    # Create log dir
    DirectoryManager(args.log_dir)
    
    return args