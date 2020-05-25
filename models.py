import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Reshape, Dropout
from tensorflow.keras.layers import Bidirectional, RepeatVector, Dot, Activation
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from kerastuner import HyperModel

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *


class StandardSeq2Seq(HyperModel):
    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        """
        Constructor for the Standard Seq2Seq Model
        
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
        Builds a standard seq2seq LSTM model. A hyperparameter determines 
        whether there will be 1 or 2 LSTM layers in the encoder.

        Arguments:
            hp -- hyperparameters object from keras-tuner

        Returns:
            model: Keras model instance
        """
        
        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm_1 = LSTM(latent_dim, return_sequences=True, 
                              name='encoder_lstm_1')
        encoder_lstm_2 = LSTM(latent_dim, return_state=True, 
                              name='encoder_lstm_2')
        decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                            name='decoder_lstm')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        
        seq_dropout = Dropout(rate=hp.Float('seq_dropout', min_value=0, max_value=0.5, 
                                            step=0.1))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', min_value=0, max_value=0.5, 
                                              step=0.1))

        # ---------------------- MODEL ------------------------
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_inputs')

        x = encoder_inputs
        
#         # This hyperparameter determines whether we should stack two LSTM layers
#         if hp.Boolean('stacked'):
#             x = encoder_lstm_1(x)
#             x = seq_dropout(x)
        
        # Obtain the hidden states of the encoder
        _, h, c = encoder_lstm_2(x)

        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_inputs')

        x = decoder_lstm(decoder_inputs, initial_state=[h, c])
        x = dense_dropout(x)
        decoder_outputs = decoder_dense(x)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, 
                                                max_value=1e-2, sampling='log'))
        model.compile(optimizer=optimizer, loss='mse')

        return model
        

def one_step_attention(encoder_outputs, h_prev, attention_repeat, 
                       attention_concatenate, attention_dense_1,
                       attention_dense_2, attention_activation,
                       attention_dot):
    """
    Performs one step of attention.
    
    Arguments:
        encoder_outputs -- outputs of the encoder, numpy-array of shape 
                           (m, Tx, 2*encoder_latent_dim)
        h_prev -- previous hidden state of the decoder LSTM, numpy-array 
                  of shape (m, decoder_latent_dim)
        attention_repeat -- predefined repeat layer
        attention_concatenate -- predefined concatenate layer
        attention_dense_1 -- predefined dense layer 
        attention_dense_2 -- predefined dense layer 
        attention_activation -- predefined activation layer
        attention_dot -- predefined dot layer
    
    Returns:
        context -- context vector, input to the decoder LSTM cell,
                   computed as dot product between the alphas and
                   the encoder outputs.
    """
    
    x = attention_repeat(h_prev)
    x = attention_concatenate([encoder_outputs, x])
    x = attention_dense_1(x)
    energies = attention_dense_2(x)
    alphas = attention_activation(energies)
    context = attention_dot([alphas, encoder_outputs])
    
    return context


class AttentiveSeq2Seq(HyperModel):
    def __init__(self, Tx, Ty, encoder_input_dim, decoder_input_dim, 
                 decoder_output_dim):
        """
        Constructor for the Attention model.
        
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
        Builds a seq2seq LSTM model with an attention mechanism. 
        The encoder LSTM layers are always BiDirectional. A hyperparameter determines 
        whether there will be 1 or 2 such layers in the encoder.

        Arguments:
            hp -- hyperparameters object from keras-tuner

        Returns:
            model: Keras model instance
        """

        # ------------------- SHARED LAYERS ---------------------
        encoder_latent_dim = hp.Int('encoder_latent_dim', min_value=32, max_value=128, 
                                    step=32)
        decoder_latent_dim = hp.Int('decoder_latent_dim', min_value=32, max_value=128, 
                                    step=32)
        attention_dense_dim = hp.Int('attention_dense_dim', min_value=8, max_value=14, 
                                     step=2)

        # Encoder layers
        encoder_lstm_1 = Bidirectional(LSTM(encoder_latent_dim, return_sequences=True, 
                                            name='encoder_lstm_1'), merge_mode='concat')
        encoder_lstm_2 = Bidirectional(LSTM(encoder_latent_dim, return_sequences=True, 
                                            name='encoder_lstm_2'), merge_mode='concat')

        # Attention layers
        attention_repeat = RepeatVector(n=self.Tx, name='attention_repeat')
        attention_concatenate = Concatenate(axis=-1, name='attention_concatenate')
        attention_dense_1 = Dense(attention_dense_dim, activation='tanh', 
                                  name='attention_dense_1')
        attention_dense_2 = Dense(1, activation='relu', name='attention_dense_2')
        attention_activation = Activation(softmax, name='attention_activation') 
        attention_dot = Dot(axes=1)

        # Decoder layers
        decoder_concatenate = Concatenate(axis=-1, name='decoder_concatenate')
        decoder_lstm = LSTM(decoder_latent_dim, return_state=True, 
                            name='decoder_lstm')
        decoder_dense = Dense(self.decoder_output_dim, activation='linear',
                              name='decoder_dense')

        seq_dropout = Dropout(rate=hp.Float('seq_dropout', min_value=0.2, max_value=0.7, 
                                            step=0.1))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', min_value=0, max_value=0.5, 
                                               step=0.1))

        # ---------------------- MODEL ------------------------
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_inputs')

        x = encoder_inputs

        # # This hyperparameter determines whether we should stack two BiLSTM layers
        # if hp.Boolean('stacked'):
        #     x = encoder_lstm_1(x)
        #     x = seq_dropout(x)

        x = encoder_lstm_2(x)
        encoder_outputs = seq_dropout(x)

        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_inputs')

        # Zeros tensors as initial values for h and c.
        # Basically, I apply the decoder LSTM on the first timestep of encoder outputs  
        # concatenated with decoder inputs in order to get the hidden states h and c, 
        # and then I create zeros tensors from their shape, because I cannot obtain 
        # the batch size dynamically. Moreover, I have to apply an identity lambda 
        # function in order to cast the zero tensor to a Keras tensor (otherwise it 
        # cannot be passed as initial_state)

        # x is a slice of the encoder outputs
        x = Lambda(lambda z: z[:, 0, :])(encoder_outputs)
        x = K.expand_dims(x, axis=1)
        # y is a slice of the decoder inputs
        y = Lambda(lambda z: z[:, 0, :])(decoder_inputs)
        y = K.expand_dims(y, axis=1)
        # Concatenate both by the last axis
        z = Concatenate(axis=-1)([x, y])
        # Feed the dummy tensor in order to obtain a sample tensor for h and c
        _, h, c = decoder_lstm(z)
        # create tensors of zeros using the shapes of the previous dummies
        h = Lambda(lambda z: z, name='h0')(K.zeros_like(h))
        c = Lambda(lambda z: z, name='c0')(K.zeros_like(c))

        # Decoder outputs
        outputs = []

        for t in range(self.Ty):
            context = one_step_attention(encoder_outputs, h, attention_repeat, 
                                         attention_concatenate, attention_dense_1,
                                         attention_dense_2, attention_activation,
                                         attention_dot)
            
            # Obtain the decoder input at timestamp t
            x = Lambda(lambda z: z[:, t, :])(decoder_inputs)
            decoder_input = K.expand_dims(x, axis=1)

            # Construct the full decoder input by concatenating the input at 
            # timestemp t with the calculated context
            full_decoder_input = decoder_concatenate([decoder_input, context])

            h, _, c = decoder_lstm(full_decoder_input, initial_state=[h, c])
            x = dense_dropout(h)
            decoder_output = decoder_dense(x)
            
            outputs.append(decoder_output)

        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, 
                                                max_value=1e-2, sampling='log'))
        model.compile(optimizer=optimizer, loss='mse')
        return model    
