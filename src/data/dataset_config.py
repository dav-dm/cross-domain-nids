from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    'iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/iomt/'
            'iomt_traffic.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'cic_iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/cic_iomt/'
            'cic-iomtl.parquet'
        ),
        'label_column': 'LABEL-bin',
    }
}
