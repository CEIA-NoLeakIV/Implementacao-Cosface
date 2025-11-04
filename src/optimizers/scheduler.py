import math
from keras.callbacks import Callback
from keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        self.model.optimizer.learning_rate.assign(lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr_tensor = getattr(self.model.optimizer, 'learning_rate', None)
        if lr_tensor is not None:
            lr_value = lr_tensor
            if hasattr(lr_tensor, 'numpy'):
                lr_value = lr_tensor.numpy()
            elif hasattr(lr_tensor, 'assign'):
                lr_value = K.get_value(lr_tensor)
            logs['lr'] = float(lr_value)
