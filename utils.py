import tensorflow.keras.backend as K
import tensorflow as tf
from datetime import datetime


class LossPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, Ty):
        self.Ty = Ty
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Prints the average of the train and validation losses of the output units 
        (in an attentive architecture).
        """
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\tepoch: {epoch}\tloss: {round(logs["loss"]/self.Ty, 5)}\tval_loss: {round(logs["val_loss"]/self.Ty, 5)}')            


def masked_mse(y_true, y_pred):
    """
    Calculates the mean square error for the outputs, by masking the imputed values.

    Arguments:
        y_true -- Keras tensor with the ground truth values, has shape (?, Ty, 2).
                  The two columns of the last dimension correspond to the pollution values 
                  and the boolean indicating missingness. 
        y_pred -- Keras tensor with the predicted values, has shape (?, Ty, 1)

    Returns:
        Float -- the calculated error
    """
    return K.mean(K.mean(((y_true[:, :, 0] - y_pred[:, :, 0]) ** 2) * (1-y_true[:, :, 1]), axis=0))


def attention_masked_mse(y_true, y_pred):
    """
    Calculates the mean square error for the outputs, by masking the imputed values.
    This form of the loss function is used in the attentive models, where each of the
    Ty outputs in the decoder is a separate unit computing its own loss.

    Arguments:
        y_true -- Keras tensor with the ground truth values, has shape (?, 2).
                  The two columns of the last dimension correspond to the pollution values 
                  and the boolean indicating missingness. 
        y_pred -- Keras tensor with the predicted values, has shape (?, 1)

    Returns:
        Float -- the calculated error
    """
    return K.mean(((y_true[:, 0] - y_pred[:, 0]) ** 2) * (1-y_true[:, 1]))


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
    