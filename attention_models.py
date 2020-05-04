import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Reshape, Dropout
from tensorflow.keras.layers import Bidirectional, RepeatVector, Dot, Activation
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from kerastuner import HyperModel

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *


def one_step_attention(encoder_outputs, h_prev, attention_repeat, 
                       attention_concatenate, attention_dense_1,
                       attention_dense_2, attention_activation,
                       attention_dot):
    """
    Performs one step of attention: Outputs a context vector computed 
    as a dot product of the attention weights "alphas" and the outputs
    of the encoder.
    
    Arguments:
    encoder_outputs -- outputs of the encoder, numpy-array of shape 
                       (m, Tx, 2*encoder_latent_dim)
    h_prev -- previous hidden state of the decoder LSTM, numpy-array 
              of shape (m, decoder_latent_dim)
    attention_repeat -- predefined repeat layer for the attention
    attention_concatenate -- predefined concatenate layer for the 
                             attention
    attention_dense_1 -- predefined dense layer for the attention
    attention_dense_2 -- predefined dense layer for the attention
    attention_activation -- predefined activation layer for the 
                            attention
    attention_dot -- predefined dot layer for the attention
    
    
    Returns:
    context -- context vector, input to the decoder LSTM cell
    """
    
    # Repeat h_prev to be of shape (m, Tx, decoder_latent_dim) 
    h_prev = attention_repeat(h_prev)
    
    # Concatenate the encoder outputs to the repeated vector
    concat = attention_concatenate([encoder_outputs, h_prev])
       
    # Compute the energies
    energies = attention_dense_1(concat)
    energies = attention_dense_2(energies)

    # Compute the alphas by applying softmax on the energies
    alphas = attention_activation(energies)

    # Compute the context vector by dotting the alphas and 
    # the encoder outputs
    context = attention_dot([alphas, encoder_outputs])
    
    return context


class AttentionModel(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, decoder_input_dim, 
                 decoder_output_dim):
        """
        Constructor for the derived HyperModel class
        
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        encoder_input_dim -- length of input vector for the encoder
        decoder_input_dim -- length of input vector for the decoder
        decoder_output_dim -- length of output vector for the decoder
        """
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim


    def build(self, hp):
        """
        Builds a seq2seq LSTM model with an attention mechanism and Teacher Forcing (TF). 

        Arguments:
        hp -- hyperparameters object from keras-tuner

        Returns:
        model -- Keras model instance
        """

        # ------------------- SHARED LAYERS ---------------------
        encoder_latent_dim = hp.Int('encoder_latent_dim', min_value=16, max_value=48, 
                                    step=16)
        decoder_latent_dim = 2 * encoder_latent_dim
        attention_dense_dim = hp.Int('attention_dense_dim', min_value=8, max_value=14, 
                                     step=2)

        # Encoder layers
        encoder_lstm = Bidirectional(LSTM(encoder_latent_dim, return_sequences=True, 
                                          name='encoder_lstm'), merge_mode='concat')

        # Attention layers
        attention_repeat = RepeatVector(self.Tx, name='attention_repeat')
        attention_concatenate = Concatenate(axis=-1, name='attention_concatenate')
        attention_dense_1 = Dense(attention_dense_dim, activation='tanh', 
                                  name='attention_dense_1')
        attention_dense_2 = Dense(1, activation='relu', name='attention_dense_2')
        attention_activation = Activation(softmax, name='attention_activation') 
        attention_dot = Dot(axes = 1)

        # Decoder layers
        decoder_concatenate = Concatenate(axis=-1, name='decoder_concatenate')
        decoder_lstm = LSTM(decoder_latent_dim, return_state=True, 
                            name='decoder_lstm')
        decoder_dense = Dense(self.decoder_output_dim, activation='linear',
                              name='decoder_dense')

        # ---------------------- MODEL ------------------------
        # Encoder inputs
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')
        encoder_outputs = encoder_lstm(encoder_inputs)

        # Decoder inputs
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Zeros tensors as initial values for h and c.
        # Basically, I apply the decoder LSTM on the first timestep of encoder outputs  
        # concatenated with decoder inputs in order to get the hidden states h and c, 
        # and then I create zeros tensors from their shape, because I cannot obtain 
        # the batch size dynamically. Moreover, I have to apply an identity lambda 
        # function in order to cast the zero tensor to a Keras tensor (otherwise it 
        # cannot be passed as initial_state)

        # x is a slice of the encoder outputs
        x = Lambda(lambda x: x[:, 0, :])(encoder_outputs)
        x = K.expand_dims(x, axis=1)
        # y is a slice of the decoder inputs
        y = Lambda(lambda y: y[:, 0, :])(decoder_inputs)
        y = K.expand_dims(y, axis=1)
        # z is a concatenation of x and y
        z = Concatenate(axis=-1)([x, y])
        # we feed the dummy tensor in order to obtain a sample tensor of h and c
        _, h, c = decoder_lstm(z)
        # create tensors of zeros using the shapes of the previous dummies
        h = Lambda(lambda x: x, name='h0')(K.zeros_like(h))
        c = Lambda(lambda x: x, name='c0')(K.zeros_like(c))

        # Decoder outputs
        outputs = []

        for t in range(self.Ty):
            # Perform attention for one timestep
            context = one_step_attention(encoder_outputs, h, attention_repeat, 
                                         attention_concatenate, attention_dense_1,
                                         attention_dense_2, attention_activation,
                                         attention_dot)
            
            # Concatenate the context vector and the decoder input
            decoder_input = Lambda(lambda x: x[:, t, :])(decoder_inputs)
            decoder_input = K.expand_dims(decoder_input, axis=1)
            full_decoder_input = decoder_concatenate([context, decoder_input])

            # Apply post-attention LSTM
            h, _, c = decoder_lstm(inputs=full_decoder_input, initial_state=[h, c])

            # Apply dense to compute the output
            decoder_output = decoder_dense(h)
            
            # Append the the model's outputs
            outputs.append(decoder_output)

        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=attention_masked_mse)
        return model