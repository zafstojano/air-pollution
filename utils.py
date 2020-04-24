import tensorflow.keras.backend as K


def masked_mse(y_true, y_pred):
    return K.sum(((y_true[:, :, 0] - y_pred[:, :, 0]) ** 2) * (1-y_true[:, :, 1]), 
                  axis=-1) / (1 + K.sum((1-y_true[:, :, 1]), axis=-1))


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
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
    