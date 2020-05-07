import pandas as pd
import numpy as np
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error

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