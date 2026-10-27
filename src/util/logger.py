import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data.dataset_config import dataset_config
from util.directory_manager import DirectoryManager


class Logger:
    def __init__(self, src_dataset_name, trg_dataset_name):
        self.log_dir = Path(DirectoryManager().log_dir).parent
        self.src_dataset_name = src_dataset_name
        self.trg_dataset_name = trg_dataset_name
        print('=' * 100)

    def _load_data(self, folder_path):
        labels_data = np.load(f'{folder_path}/labels.npz')
        preds_data = np.load(f'{folder_path}/preds.npz')
        labels = labels_data[labels_data.files[0]]
        preds = preds_data[preds_data.files[0]]
        return labels, preds

    def _compute_metrics(self, labels, preds):
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'precision_micro': precision_score(labels, preds, average='micro', zero_division=0),
            'recall_micro': recall_score(labels, preds, average='micro', zero_division=0),
            'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
            'classification_report': classification_report(labels, preds, zero_division=0),
        }

    def _save_report(self, metrics, folder_name, path):
        path.mkdir(parents=True, exist_ok=True)
        report_path = path / f'report_{folder_name}.txt'
        
        with report_path.open('w') as f:
            f.write('Metrics Report\n')
            f.write('=========================\n')
            for key, value in metrics.items():
                if key == 'classification_report':
                    f.write(f'\n{key}:\n{value}\n')
                else:
                    f.write(f'{key}: {value}\n')
        print(f'Report saved => {report_path}')

    def _generate_confusion_matrices(self, labels, preds, folder_name, path, task):
        path.mkdir(parents=True, exist_ok=True)
        
        cm = confusion_matrix(labels, preds)
        np.savetxt(path / f'confusion_matrix_{folder_name}.csv', cm, delimiter=',', fmt='%d')
        cm_norm = confusion_matrix(labels, preds, normalize='true')

        dataset_name = self.src_dataset_name if task == 'src' else self.trg_dataset_name
        dc = dataset_config[dataset_name]
        label_column = dc.get('label_column', 'label').lower()
        label_conv_path = Path(dc['path']).parent / f'{label_column}_conv.json'
        with label_conv_path.open('r') as f:
            label_conv = json.load(f)

        classes = label_conv.keys()
        n_classes = len(classes)

        plt.figure(figsize=(max(6, n_classes * 0.8), max(4, n_classes * 0.6))) # Dynamic fig size
        sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='viridis')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(path / f'confusion_matrix_{folder_name}.pdf', bbox_inches='tight')
        plt.close()

    def process_results(self):
        """
        Process results for both source (src) and target (trg) tasks in each phase (train, val, test).
        - Loads labels and predictions.
        - Computes metrics and saves them to a text file.
        - Generates confusion matrix CSV and a normalized confusion matrix plot.
        """
        for task in ['src', 'trg']:
            base_path = self.log_dir / task
            if not base_path.exists():
                continue

            for phase in ['train', 'val', 'test']:
                path = base_path / phase
                if not path.exists():
                    continue

                labels, preds = self._load_data(path)
                metrics = self._compute_metrics(labels, preds)
                self._save_report(metrics, folder_name=phase, path=base_path / 'detailed')
                self._generate_confusion_matrices(labels, preds, folder_name=phase, path=base_path / 'detailed', task=task)
                

    def _plot_metric_group(self, df, group_cols, ylabel, path, output_filename):
        max_epoch = df.shape[0]
        plt.figure(figsize=(max_epoch * 0.20, 5))
        for col in group_cols:
            plt.plot(df['epoch'], df[col], marker='o', label=col)

        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid(linestyle='--', color='gray')
        plt.xticks(np.arange(1, max_epoch + 1, 1), fontsize=8, rotation=90)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.savefig(path / output_filename, bbox_inches='tight')
        plt.close()

    def plot_per_epoch_metrics(self, filename='epoch_metrics'):
        """
        Plot per-epoch metrics for both source (src) and target (trg) tasks.
        The data should be stored in a Parquet file named `filename` (default: 'epoch_metrics').
        """
        for task in ['src', 'trg']:
            base_path = self.log_dir / task
            if not base_path.exists():
                continue
        
            file_path = Path(base_path) / f'{filename}.csv'
            if not file_path.exists():
                continue
            
            df = pd.read_csv(file_path)

            metrics_columns = [col for col in df.columns if col != 'epoch']
            loss_cols = [col for col in metrics_columns if 'loss' in col.lower()]
            other_cols = [col for col in metrics_columns if col not in loss_cols]

            self._plot_metric_group(
                df, loss_cols, ylabel='Loss Value', 
                path=base_path / 'detailed', output_filename='epoch_loss.pdf'
            )
            self._plot_metric_group(
                df, other_cols, ylabel='Metric Value', 
                path=base_path / 'detailed', output_filename='epoch_metrics.pdf'
            )
