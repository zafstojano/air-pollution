import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Reshape, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from kerastuner import HyperModel
from utils import *


class SimpleSeq2Seq(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim

        
    def build(self, hp):

        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
        decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                            return_state=True, name='decoder_lstm')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', 0, 0.7, 
                                step=0.1, default=0.5))
        
        
        # ---------------------- MODEL ------------------------
        # Define the inputs for the encoder
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')

        # We discard output and keep the states only.
        _, h, c = encoder_lstm(encoder_inputs)

        # Define an input for the decoder
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Obtain all the outputs from the decoder (return_sequences = True)
        decoder_outputs, _, _  = decoder_lstm(decoder_inputs, initial_state=[h, c])

        # Apply dropout
        decoder_outputs = dense_dropout(decoder_outputs)

        # Apply dense 
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=masked_mse)

        return model
        

class StackedEncoderSeq2Seq(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim

        
    def build(self, hp):

        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm_1 = LSTM(latent_dim, return_sequences=True, 
                              name='encoder_lstm_1')
        encoder_lstm_2 = LSTM(latent_dim, return_state=True, 
                              name='encoder_lstm_2')
        decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                            return_state=True, name='decoder_lstm')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        
        seq_dropout = Dropout(rate=hp.Float('seq_dropout', 0, 0.7, 
                                step=0.1, default=0.5))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', 0, 0.7, 
                                step=0.1, default=0.5))

        
        # ---------------------- MODEL ------------------------
        # Define the inputs for the encoder
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')
        
        # Obtain the outputs from the first encoder layer
        encoder_out = encoder_lstm_1(encoder_inputs) 
        
        # Pass the outputs through a dropout layer before 
        # feeding them to the next LSTM layer
        encoder_out = seq_dropout(encoder_out)
        
        # We discard the output and keep the states only.
        _, h, c = encoder_lstm_2(encoder_out)

        # Define an input for the decoder
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Obtain all the outputs from the decoder (return_sequences = True)
        decoder_outputs, _, _  = decoder_lstm(decoder_inputs, initial_state=[h, c])

        # Apply dropout
        decoder_outputs = dense_dropout(decoder_outputs)

        # Apply dense 
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=masked_mse)

        return model
        

class BiStackedEncoderSeq2Seq(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim

        
    def build(self, hp):

        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm_1 = Bidirectional(LSTM(latent_dim, return_sequences=True, 
                              name='encoder_lstm_1'), merge_mode='concat')
        encoder_lstm_2 = LSTM(latent_dim, return_state=True, 
                              name='encoder_lstm_2')
        decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                            return_state=True, name='decoder_lstm')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        seq_dropout = Dropout(rate=hp.Float('seq_dropout', 0, 0.7, 
                                step=0.1, default=0.5))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', 0, 0.7, 
                                step=0.1, default=0.5))

        
        # ---------------------- MODEL ------------------------
        # Define the inputs for the encoder
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')
        
        # Obtain the outputs from the first encoder layer
        encoder_out = encoder_lstm_1(encoder_inputs) 
        
        # Pass the outputs through a dropout layer before 
        # feeding them to the next LSTM layer
        encoder_out = seq_dropout(encoder_out)
        
        # We discard the output and keep the states only.
        _, h, c = encoder_lstm_2(encoder_out)

        # Define an input for the decoder
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Obtain all the outputs from the decoder (return_sequences = True)
        decoder_outputs, _, _  = decoder_lstm(decoder_inputs, initial_state=[h, c])

        # Apply dropout
        decoder_outputs = dense_dropout(decoder_outputs)

        # Apply dense 
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=masked_mse)

        return model


class StackedDecoderSeq2Seq(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim

        
    def build(self, hp):

        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm = LSTM(latent_dim, return_state=True, 
                              name='encoder_lstm')
        decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, 
                              name='decoder_lstm_1')
        decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, 
                            return_state=True, name='decoder_lstm_2')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        seq_dropout = Dropout(rate=hp.Float('seq_dropout', 0, 0.7, 
                                step=0.1, default=0.5))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', 0, 0.7, 
                                step=0.1, default=0.5))

        
        # ---------------------- MODEL ------------------------
        # Define the inputs for the encoder
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')
        
        # We discard the output and keep the states only.
        _, h, c = encoder_lstm(encoder_inputs)

        # Define an input for the decoder
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Obtain all the outputs from the decoder (return_sequences = True)
        decoder_outputs = decoder_lstm_1(decoder_inputs, initial_state=[h, c])

        # Apply dropout
        decoder_outputs = seq_dropout(decoder_outputs)

        # Apply LSTM again (stacked)
        decoder_outputs, _, _ = decoder_lstm_2(decoder_outputs)
        
        # Apply dropout
        decoder_outputs = dense_dropout(decoder_outputs)
        
        # Apply dense 
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=masked_mse)

        return model
        

class StackedSeq2Seq(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim

        
    def build(self, hp):

        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm_1 = LSTM(latent_dim, return_sequences=True,
                              name='encoder_lstm_1')
        encoder_lstm_2 = LSTM(latent_dim, return_state=True, 
                              name='encoder_lstm_2')
        decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, 
                              name='decoder_lstm_1')
        decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, 
                              return_state=True, name='decoder_lstm_2')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        seq_dropout = Dropout(rate=hp.Float('seq_dropout', 0, 0.7, 
                                step=0.1, default=0.5))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', 0, 0.7, 
                                step=0.1, default=0.5))

        
        # ---------------------- MODEL ------------------------
        # Define the inputs for the encoder
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')
        
        # First layer in the encoder
        encoder_outputs = encoder_lstm_1(encoder_inputs)
        
        # Apply dropout
        encoder_outputs = seq_dropout(encoder_outputs)
        
        # Pass the outputs to the next encoder layer, obtain h and c
        _, h, c = encoder_lstm_2(encoder_outputs)

        # Define an input for the decoder
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Obtain all the outputs from the decoder (return_sequences = True)
        decoder_outputs = decoder_lstm_1(decoder_inputs, initial_state=[h, c])

        # Apply dropout
        decoder_outputs = seq_dropout(decoder_outputs)

        # Apply LSTM again (stacked)
        decoder_outputs, _, _ = decoder_lstm_2(decoder_outputs)
        
        # Apply dropout
        decoder_outputs = dense_dropout(decoder_outputs)
        
        # Apply dense 
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=masked_mse)

        return model
        

class BiStackedSeq2Seq(HyperModel):

    def __init__(self, Tx, Ty, encoder_input_dim, 
                 decoder_input_dim, decoder_output_dim):
        self.Tx = Tx
        self.Ty = Ty
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.decoder_output_dim = decoder_output_dim

        
    def build(self, hp):

        # ------------------- SHARED LAYERS ---------------------
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=128, step=32)
        
        encoder_lstm_1 = Bidirectional(LSTM(latent_dim, return_sequences=True,
                              name='encoder_lstm_1'), merge_mode='concat')
        encoder_lstm_2 = LSTM(latent_dim, return_state=True, 
                              name='encoder_lstm_2')
        decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, 
                              name='decoder_lstm_1')
        decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, 
                              return_state=True, name='decoder_lstm_2')
        decoder_dense = Dense(self.decoder_output_dim, 
                              activation='linear', name='decoder_dense')
        seq_dropout = Dropout(rate=hp.Float('seq_dropout', 0, 0.7, 
                                step=0.1, default=0.5))
        dense_dropout = Dropout(rate=hp.Float('dense_dropout', 0, 0.7, 
                                step=0.1, default=0.5))

        
        # ---------------------- MODEL ------------------------
        # Define the inputs for the encoder
        encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), 
                               name='encoder_input')
        
        # First layer in the encoder
        encoder_outputs = encoder_lstm_1(encoder_inputs)
        
        # Apply dropout
        encoder_outputs = seq_dropout(encoder_outputs)
        
        # Pass the outputs to the next encoder layer, obtain h and c
        _, h, c = encoder_lstm_2(encoder_outputs)

        # Define an input for the decoder
        decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), 
                               name='decoder_input')

        # Obtain all the outputs from the decoder (return_sequences = True)
        decoder_outputs = decoder_lstm_1(decoder_inputs, initial_state=[h, c])

        # Apply dropout
        decoder_outputs = seq_dropout(decoder_outputs)

        # Apply LSTM again (stacked)
        decoder_outputs, _, _ = decoder_lstm_2(decoder_outputs)
        
        # Apply dropout
        decoder_outputs = seq_dropout(decoder_outputs)
        
        # Apply dense 
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], 
                      outputs=decoder_outputs)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, 
                                                sampling='log'))
        model.compile(optimizer=optimizer, loss=masked_mse)

        return model
        