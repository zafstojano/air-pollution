import tensorflow.keras.backend as K
import tensorflow as tf
from datetime import datetime


class LossPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, Ty):
        self.Ty = Ty
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Prints the average train and val losses (in an attentive architecture).
        """
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\tepoch: {epoch}\tloss: {round(logs["loss"]/self.Ty, 5)}\tval_loss: {round(logs["val_loss"]/self.Ty, 5)}')            


def softmax(x, axis=1):
    """Softmax activation function.
    Arguments:
        x: Tensor.
        axis: Integer -- axis along which the softmax normalization is applied.
    Returns:
        Tensor -- output of softmax transformation.
    Raises:
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
    