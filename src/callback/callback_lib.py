class Callback:
    """
    Base class for Callbacks.
    """
    
    def on_fit_start(self, module):
        # Called once, at the beginning of the fit.
        pass

    def on_fit_end(self, module):
        # Called once, at the end of the fit.
        pass

    def on_validation_start(self, module):
        # Called at the start of validation (separate from training).
        pass

    def on_validation_end(self, module):
        # Called at the end of validation.
        pass

    def on_test_start(self, module):
        # Called at the beginning of the test.
        pass

    def on_test_end(self, module):
        # Called at the end of the test.
        pass
    
    def on_adaptation_start(self, module):
        # Called once, at the beginning of the adaptation.
        pass

    def on_adaptation_end(self, module):
        # Called once, at the end of the adaptation.
        pass
    
    def on_epoch_start(self, module, epoch):
        # Called at the beginning of a fit epoch
        pass

    def on_epoch_end(self, module, epoch):
        # Called at the end of a fit epoch
        pass