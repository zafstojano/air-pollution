import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime, timedelta
import os
from data_utils import *
from config import best_hyperparameters
import pickle
import pytz

"""
	This script should generate predictions for each sensor and store them in a DB.
	For now, it writes them in a 5 separate CSV files.
"""

# Pull the necessary data for inference
inference_data_pipeline()

pipeline_type = PIPELINE_PREDICTIONS

if not os.path.exists(f'./data/{pipeline_type}'):
	os.makedirs(f'./data/{pipeline_type}')

# Generate the predictions for each SENSORS
for station, pollutants in SENSORS.items():
	print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tGenerating predictions for {station}')
	model = load_model(f'./inference-models/{station}.h5')

	encoder_input_data = np.load(f'./data/inference-encoder/third-order/{station}/encoder_input_data.npy')
	decoder_input_pollution_data = np.load(f'./data/inference-decoder/third-order/{station}/decoder_input_pollution_data.npy')
	decoder_input_extras_data = np.load(f'./data/inference-decoder/third-order/{station}/decoder_input_extras_data.npy')

	hp = best_hyperparameters[station]
	decoder_latent_dim = hp['decoder_latent_dim']
	h0 = np.zeros((1, decoder_latent_dim))
	c0 = np.zeros((1, decoder_latent_dim))

	pred = model.predict([encoder_input_data, decoder_input_pollution_data, decoder_input_extras_data, h0, c0])

	# reshape data
	pred = np.array(pred)
	pred = np.swapaxes(pred, 0, 1)
	pred = np.squeeze(pred, axis=0)

	# retransform data
	for i, pollutant in enumerate(pollutants):
		with open(f'./pickles/scalers/{station}/{pollutant}', 'rb') as f:
			scaler = pickle.load(f)
			pred[:, i] = np.exp(scaler.inverse_transform(pred[:, i]) - 1)

	# If any of the predictions are negative, set them to 0
	pred[pred < 0] = 0

	# generate future timesteps for the predictions
	now = datetime.now(TZ_MKD)
	now = TZ_MKD.localize(datetime(now.year, now.month, now.day, now.hour, 0, 0))
	Ty = pred.shape[0]
	future_timesteps = [now + timedelta(hours=i) for i in range(1, Ty+1)]

	# save the predictions to a CSV file
	with open(f'./data/{pipeline_type}/{station}.csv', 'w') as f:
		f.write('Timestamp,' + ",".join(pollutants) + "\n")
		for i, timestamp in enumerate(future_timesteps):
			f.write(timestamp.strftime("%Y-%m-%d, %H:%M:%S") + ',' + ",".join(str(item) for item in pred[i, :]) + '\n')
