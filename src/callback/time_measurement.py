import time
import json

from callback.callback_lib import Callback
from util.directory_manager import DirectoryManager


class TimeMeasurement(Callback):
    """
    Callback that measures overall training and testing time.
    """
    def on_fit_start(self, _):
        self.train_start_time = time.time()

    def on_fit_end(self, _):
        self.measurements = {'train_time': time.time() - self.train_start_time}
        
    def on_adaptation_start(self, _):
        self.train_start_time = time.time()

    def on_adaptation_end(self, _):
        self.measurements = {'train_time': time.time() - self.train_start_time}

    def on_test_start(self, _):
        self.test_start_time = time.time()

    def on_test_end(self, _):
        if not hasattr(self, 'measurements'):
            self.measurements = {}
        self.measurements['test_time'] = time.time() - self.test_start_time

        detailed_dir = DirectoryManager().mkdir('detailed')
        with open(f'{detailed_dir}/time_measurements.json', 'w') as f:
            json.dump(self.measurements, f)
