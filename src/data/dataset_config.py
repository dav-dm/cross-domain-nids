from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    'iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/iomt/'
            'iomt_traffic_data_100pkts_7f_load_net_no-dupes-wIAT_no-pl-mt1500_no-load_clean-uniform.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'cic_iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/cic_iomt/'
            'cic-iomt_100pkts_7f_1p-mt1m_10p-mt100k_no-dupes-wIAT_no-load_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL-bin',
    }
}
