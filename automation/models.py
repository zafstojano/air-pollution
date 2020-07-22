import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Reshape, Dropout
from tensorflow.keras.layers import Bidirectional, RepeatVector, Dot, Activation
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from kerastuner import HyperModel

import numpy as np
import pandas as pd
from utils import *


def one_step_attention(encoder_outputs, h_prev, attention_repeat, 
					   attention_concatenate, attention_dense_1,
					   attention_dense_2, attention_softmax,
					   attention_dot):
	"""
		Performs one step of attention in the decoding sequence.
		
		Arguments:
			encoder_outputs -- outputs of the encoder, tensor of shape 
							   (?, Tx, 2*encoder_latent_dim)
			h_prev -- previous hidden state of the decoder LSTM, tensor 
					  of shape (?, decoder_latent_dim)
			attention_repeat -- predefined repeat layer
			attention_concatenate -- predefined concatenate layer
			attention_dense_1 -- predefined dense layer 
			attention_dense_2 -- predefined dense layer 
			attention_softmax -- predefined softmax layer
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
	alphas = attention_softmax(energies)
	context = attention_dot([alphas, encoder_outputs])
	
	return context


class HyperAttentiveSeq2Seq(HyperModel):
	def __init__(self, Tx, Ty, encoder_input_dim, decoder_input_dim, 
				 decoder_output_dim):
		"""
			Constructor for an attention model used during the random search of
			best hyperparameters.
			
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

			Arguments:
				hp -- hyperparameters object from keras-tuner

			Returns:
				model: Keras model instance
		"""

		# ------------------- HYPERPARAMETERS ---------------------
		encoder_latent_dim = hp.Int('encoder_latent_dim', min_value=64, max_value=128, step=32)
		decoder_latent_dim = hp.Int('decoder_latent_dim', min_value=64, max_value=128, step=32)
		attention_dense_dim = hp.Int('attention_dense_dim', min_value=10, max_value=16, step=2)
		seq_dropout_rate = hp.Float('seq_dropout_rate', min_value=0, max_value=0.5, step=0.1)
		dense_dropout_rate = hp.Float('dense_dropout_rate', min_value=0, max_value=0.5, step=0.1)
		learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

		# ------------------- SHARED LAYERS ---------------------
		# Encoder layers
		encoder_lstm = Bidirectional(LSTM(encoder_latent_dim, return_sequences=True, 
											name='encoder_lstm'), merge_mode='concat')
		seq_dropout = Dropout(rate=seq_dropout_rate, name='seq_dropout')

		# Attention layers
		attention_repeat = RepeatVector(n=self.Tx, name='attention_repeat')
		attention_concatenate = Concatenate(axis=-1, name='attention_concatenate')
		attention_dense_1 = Dense(attention_dense_dim, activation='tanh', name='attention_dense_1')
		attention_dense_2 = Dense(1, activation='relu', name='attention_dense_2')
		attention_softmax = Activation(softmax, name='attention_softmax') 
		attention_dot = Dot(axes=1, name='attention_dot')

		# Decoder layers
		decoder_concatenate = Concatenate(axis=-1, name='decoder_concatenate')
		decoder_lstm = LSTM(decoder_latent_dim, return_state=True, name='decoder_lstm')
		dense_dropout = Dropout(rate=dense_dropout_rate, name='dense_dropout')
		decoder_dense = Dense(self.decoder_output_dim, activation='linear', name='decoder_dense')

		# ---------------------- MODEL ------------------------
		encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), name='encoder_inputs')
		decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), name='decoder_inputs')

		x = encoder_lstm(encoder_inputs)
		encoder_outputs = seq_dropout(x)

		"""
		Zeros tensors as initial values for h and c.
		Basically, I apply the decoder LSTM on the first timestep of encoder outputs  
		concatenated with decoder inputs in order to get the hidden states h and c, 
		and then I create zeros tensors from their shape, because I cannot obtain 
		the batch size dynamically. Moreover, I have to apply an identity lambda 
		function in order to cast the zeros tensor to a Keras tensor (otherwise it 
		cannot be passed as initial_state)
		"""
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
										 attention_dense_2, attention_softmax,
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
		optimizer = Adam(learning_rate=learning_rate)
		model.compile(optimizer=optimizer, loss='mse')
		return model    


class AttentiveSeq2SeqModelBuilder:
	def __init__(self, Tx, Ty, encoder_input_dim, decoder_input_dim, 
				 decoder_output_dim, num_pollutants, encoder_latent_dim, 
				 decoder_latent_dim, attention_dense_dim, seq_dropout_rate,
				 dense_dropout_rate, learning_rate):
		"""
			Constructor for a model builder that provides attentive seq2seq models.
			
			Arguments:
				Tx -- length of the input sequence
				Ty -- length of the output sequence
				encoder_input_dim -- length of input vector for the encoder
				decoder_input_dim -- length of input vector for the decoder
				decoder_output_dim -- length of output vector for the decoder
				num_pollutants -- number of pollutants the model is predicting
				encoder_latent_dim -- best hyperparameter value for encoder_latent_dim
				decoder_latent_dim -- best hyperparameter value for decoder_latent_dim 
				attention_dense_dim -- best hyperparameter value for attention_dense_dim 
				seq_dropout_rate -- best hyperparameter value for seq_dropout_rate 
				dense_dropout_rate -- best hyperparameter value for dense_dropout_rate 
				learning_rate -- best hyperparameter value for learning_rate 
		"""

		self.Tx = Tx
		self.Ty = Ty
		self.encoder_input_dim = encoder_input_dim
		self.decoder_input_dim = decoder_input_dim
		self.decoder_output_dim = decoder_output_dim
		self.num_pollutants = num_pollutants
		self.encoder_latent_dim = encoder_latent_dim
		self.decoder_latent_dim = decoder_latent_dim
		self.attention_dense_dim = attention_dense_dim
		self.seq_dropout_rate = seq_dropout_rate
		self.dense_dropout_rate = dense_dropout_rate
		self.learning_rate = learning_rate


	def build_training_attentive_model(self):
		"""
			Builds an attentive model for training given best found hyperparameters. 

			Returns:
				model: Keras model instance
		"""
		K.clear_session()

		# ------------------- SHARED LAYERS ---------------------
		# Encoder layers
		encoder_lstm = Bidirectional(LSTM(self.encoder_latent_dim, return_sequences=True, 
										  name='encoder_lstm'), merge_mode='concat')
		seq_dropout = Dropout(rate=self.seq_dropout_rate, name='seq_dropout')

		# Attention layers
		attention_repeat = RepeatVector(n=self.Tx, name='attention_repeat')
		attention_concatenate = Concatenate(axis=-1, name='attention_concatenate')
		attention_dense_1 = Dense(self.attention_dense_dim, activation='tanh', name='attention_dense_1')
		attention_dense_2 = Dense(1, activation='relu', name='attention_dense_2')
		attention_softmax = Activation(softmax, name='attention_softmax') 
		attention_dot = Dot(axes=1, name='attention_dot')

		# Decoder layers
		decoder_concatenate = Concatenate(axis=-1, name='decoder_concatenate')
		decoder_lstm = LSTM(self.decoder_latent_dim, return_state=True, name='decoder_lstm')
		dense_dropout = Dropout(rate=self.dense_dropout_rate, name='dense_dropout')
		decoder_dense = Dense(self.decoder_output_dim, activation='linear', name='decoder_dense')

		# -------------------- TRAIN MODEL --------------------
		encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), name='encoder_inputs')
		decoder_inputs = Input(shape=(self.Ty, self.decoder_input_dim), name='decoder_inputs')

		x = encoder_lstm(encoder_inputs)
		encoder_outputs = seq_dropout(x)

		h0 = Input(shape=(self.decoder_latent_dim,), name='h0')
		c0 = Input(shape=(self.decoder_latent_dim,), name='c0')
		h, c = h0, c0

		# Decoder outputs
		outputs = []

		for t in range(self.Ty):
			context = one_step_attention(encoder_outputs, h, attention_repeat, 
										 attention_concatenate, attention_dense_1,
										 attention_dense_2, attention_softmax,
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

		model = Model(inputs=[encoder_inputs, decoder_inputs, h0, c0], outputs=outputs)
		optimizer = Adam(learning_rate=self.learning_rate)
		model.compile(optimizer=optimizer, loss='mse')
		return model


	def build_inference_attentive_model(self, best_model):
		"""
			Builds an attentive model for inference given pretrained layers
			of the best performing model. 

			Arguments:
				best_model: Keras model instance

			Returns:
				Keras model instance
		"""
		K.clear_session()

		# ------------------- SHARED LAYERS ---------------------
		# Encoder layers
		encoder_lstm = best_model.get_layer('bidirectional')
		seq_dropout = best_model.get_layer('seq_dropout')

		# Attention layers
		attention_repeat = best_model.get_layer('attention_repeat')
		attention_concatenate = best_model.get_layer('attention_concatenate')
		attention_dense_1 = best_model.get_layer('attention_dense_1')
		attention_dense_2 = best_model.get_layer('attention_dense_2')
		attention_softmax = best_model.get_layer('attention_softmax')
		attention_dot = best_model.get_layer('attention_dot')

		# Decoder layers
		decoder_concatenate = best_model.get_layer('decoder_concatenate')
		decoder_lstm = best_model.get_layer('decoder_lstm')
		dense_dropout = best_model.get_layer('dense_dropout')
		decoder_dense = best_model.get_layer('decoder_dense')

		# -------------------- INFERENCE MODEL --------------------
		encoder_inputs = Input(shape=(self.Tx, self.encoder_input_dim), name='encoder_inputs')
		decoder_pollutants_input = Input(shape=(self.num_pollutants,), name='decoder_pollutants_input')
		decoder_extras_inputs = Input(shape=(self.Ty, self.decoder_input_dim - self.num_pollutants), 
									  name='decoder_extras_inputs')

		x = encoder_lstm(encoder_inputs)
		encoder_outputs = seq_dropout(x)

		h0 = Input(shape=(self.decoder_latent_dim,), name='h0')
		c0 = Input(shape=(self.decoder_latent_dim,), name='c0')
		h, c = h0, c0

		# Decoder outputs
		outputs = []

		for t in range(self.Ty):
			context = one_step_attention(encoder_outputs, h, attention_repeat, 
										 attention_concatenate, attention_dense_1,
										 attention_dense_2, attention_softmax,
										 attention_dot)
			
			# Obtain the decoder input extras at timestamp t
			decoder_extras_input = Lambda(lambda z: z[:, t, :])(decoder_extras_inputs)
			
			# If it's the first timestep, grab the pollutant values passed as input,
			# otherwise grab the predictions in the previous step
			if t == 0:
				x = decoder_pollutants_input
			else:
				x = decoder_output

			x = Concatenate(axis=-1)([x, decoder_extras_input])
			decoder_input = K.expand_dims(x, axis=1)

			# Construct the full decoder input by concatenating the input at 
			# timestemp t with the calculated context
			full_decoder_input = decoder_concatenate([decoder_input, context])

			h, _, c = decoder_lstm(full_decoder_input, initial_state=[h, c])
			x = dense_dropout(h)
			decoder_output = decoder_dense(x)
			outputs.append(decoder_output)

		return Model(inputs=[encoder_inputs, decoder_pollutants_input, decoder_extras_inputs, h0, c0], 
					 outputs=outputs)
