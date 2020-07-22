import requests
import time
from datetime import datetime, timedelta
from copy import deepcopy
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import pickle
import pytz
import shutil

# Remove the line below if you want to run the code, this is a hidden file with API keys
from keys import *

TZ_MKD = pytz.timezone('Europe/Skopje')

SENSORS = {
	'Miladinovci': ['PM10', 'O3', 'CO', 'NO2', 'SO2'],
	'Rektorat': ['PM10', 'O3', 'CO'],
	'Lisice': ['PM10', 'O3', 'CO', 'SO2'],
	'Centar': ['PM10', 'PM25'],
	'Karpos': ['PM10', 'PM25']  
}

SENSOR_NOMINAL_OPERABILITY = {
	'Miladinovci': TZ_MKD.localize(datetime(2016, 4, 14, 12, 0, 0)),
	'Rektorat': TZ_MKD.localize(datetime(2014, 7, 30, 17, 0, 0)),
	'Lisice': TZ_MKD.localize(datetime(2012, 12, 17, 15, 0, 0)),
	'Centar': TZ_MKD.localize(datetime(2011, 9, 13, 18, 0, 0)),
	'Karpos': TZ_MKD.localize(datetime(2011, 9, 14, 12, 0, 0))
}

PIPELINE_TRAINING = 'training'
PIPELINE_INFERENCE_ENCODER = 'inference-encoder'
PIPELINE_INFERENCE_DECODER = 'inference-decoder'
PIPELINE_PREDICTIONS = 'predictions'


def fetch_air_pollution(date_start, date_end, pipeline_type):
	"""
		This utility function fetches the air pollution data for all 5 government sensors. 
		The formatted results are stored in './data/pipeline_type/formatted-sensor'.
		
		Arguments:
			date_start -- beginning of fetching period
			date_end -- ending of fetching period
			pipeline_type -- type of data pipeline (possible values: PIPELINE_TRAINING, 
								PIPELINE_INFERENCE_ENCODER, PIPELINE_INFERENCE_DECODER)
	"""
	period=timedelta(days=183)
	offset=timedelta(seconds=1)
	sleep_penalty_time = 5

	if not os.path.exists(f'./data/{pipeline_type}/raw-sensor'):
		os.makedirs(f'./data/{pipeline_type}/raw-sensor')

	if not os.path.exists(f'./data/{pipeline_type}/formatted-sensor'):
		os.makedirs(f'./data/{pipeline_type}/formatted-sensor')

	for station, pollutants in SENSORS.items():
		dataframes = []
		for pollutant in pollutants:
			results = []
			if pipeline_type == PIPELINE_TRAINING:
				curr_start = deepcopy(SENSOR_NOMINAL_OPERABILITY[station])
			else:
				curr_start = deepcopy(date_start)

			while curr_start < date_end:
				try:
					time.sleep(1)
					curr_end = min(curr_start + period - offset, date_end)

					print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\t{curr_start.strftime("%Y-%m-%d %H:%M:%S")} - {curr_end.strftime("%Y-%m-%d %H:%M:%S")}\t{station} {pollutant}')

					URL = f'http://air.moepp.gov.mk/graphs/site/pages/MakeGraph.php?station={station}'+\
					  f'&parameter={pollutant}'+\
					  f'&beginDate={curr_start.strftime("%Y-%m-%d")}'+\
					  f'&beginTime={curr_start.strftime("%H:%M")}'+\
					  f'&endDate={curr_end.strftime("%Y-%m-%d")}'+\
					  f'&endTime={curr_end.strftime("%H:%M")}'+\
					  f'&i={MOEPP_KEY}&lang=mk'

					r = requests.get(url=URL) 
					data = r.json()

					measurements = data['measurements'][0]['data']
					timestamps = data['times']

					rows = list(zip(timestamps, measurements))
					results.extend(rows)
					
					# reset penalty for waiting
					sleep_penalty_time = 5

					curr_start += period
				
				except Exception as e:
					print(e)
					time.sleep(sleep_penalty_time)
					# Increase penalty in case of an exception
					sleep_penalty_time += 5

					# If we keep throwing multiple consecutive exceptions, terminate
					if sleep_penalty_time > 30:
						raise Exception('Cannot fetch air pollution data')
			
			with open(f'./data/{pipeline_type}/raw-sensor/{station}_{pollutant}', 'w') as f:
				f.write('Timestamp,Value\n')
				for timestamp, value in results:
					f.write(f'{timestamp},{value}\n')
			
			# set index column, rename value column and replace 'None' with np.nan
			df = pd.read_csv(f'./data/{pipeline_type}/raw-sensor/{station}_{pollutant}')
			df.Value = df.Value.apply(lambda x: np.float32(x) if x != 'None' else np.nan)
			df = df.set_index('Timestamp')
			df = df.rename(columns={"Value": pollutant})
			dataframes.append(df)

		# concat all pollutants for a single station and save as a single pandas dataframe
		df = pd.concat(dataframes, axis=1)
		df.to_csv(f'./data/{pipeline_type}/formatted-sensor/{station}', index=True)


def fetch_datetime(date_start, date_end, pipeline_type):
	"""
		This utility function generates datetime features for the given period.
		
		Arguments:
			date_start -- beginning of generating period
			date_end -- ending of generating period
			pipeline_type -- type of data pipeline (possible values: PIPELINE_TRAINING, 
							 PIPELINE_INFERENCE_ENCODER, PIPELINE_INFERENCE_DECODER)
	"""
	date_start = TZ_MKD.localize(datetime(date_start.year, date_start.month, date_start.day, date_start.hour, 0, 0))
	date_end = TZ_MKD.localize(datetime(date_end.year, date_end.month, date_end.day, date_end.hour, 0, 0))

	HOLIDAYS = ['01-01', '06-01', '07-01', '01-05', '24-05', '02-08', '08-09', '11-10', '25-10', '08-12']
	period=timedelta(hours=1)
	curr_date = deepcopy(date_start)
	data = {}

	print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\t{date_start.strftime("%Y-%m-%d %H:%M:%S")} - {date_end.strftime("%Y-%m-%d %H:%M:%S")}\tdatetime')

	while curr_date <= date_end:
		timestamp = curr_date.strftime("%Y-%m-%d %H:%M:%S")
		hour_of_day = curr_date.hour
		day_of_week = curr_date.weekday()
		month_of_yr = curr_date.month - 1
		
		hour_sin = np.sin(2*np.pi*hour_of_day/24)
		hour_cos = np.cos(2*np.pi*hour_of_day/24)
		weekday_sin = np.sin(2*np.pi*day_of_week/7)
		weekday_cos = np.cos(2*np.pi*day_of_week/7)
		month_sin = np.sin(2*np.pi*month_of_yr/7)
		month_cos = np.cos(2*np.pi*month_of_yr/7)
		
		weekend = 1 if day_of_week == 5 or day_of_week == 6 else 0
		holiday = 1 if curr_date.strftime("%d-%m") in HOLIDAYS else 0
		
		data[timestamp] = (hour_sin, hour_cos, weekday_sin, weekday_cos, 
						   month_sin, month_cos, weekend, holiday)
		
		curr_date += period

	df = pd.DataFrame.from_dict(data, orient='index', 
								columns=['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
										 'month_sin', 'month_cos', 'weekend', 'holiday'])

	df.index.rename('Timestamp', inplace=True)

	if not os.path.exists(f'./data/{pipeline_type}'):
		os.makedirs(f'./data/{pipeline_type}')

	df.to_csv(f'./data/{pipeline_type}/datetime', index=True)


def fetch_historic_weather(date_start, date_end, pipeline_type):
	"""
		This utility function fetches historical weather data for the specified period.
		
		Arguments:
			date_start -- beginning of fetching period
			date_end -- ending of fetching period
			pipeline_type -- type of data pipeline (possible values: PIPELINE_TRAINING, PIPELINE_INFERENCE_ENCODER)
	"""
	period=timedelta(days=31)
	offset=timedelta(seconds=1)
	sleep_penalty_time = 5

	results = []
	curr_start = deepcopy(date_start)
	while curr_start < date_end:
		try:
			time.sleep(1)
			curr_end = min(curr_start + period - offset, date_end)
			print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\t{curr_start.strftime("%Y-%m-%d %H:%M:%S")} - {curr_end.strftime("%Y-%m-%d %H:%M:%S")}\tweather')

			URL = f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?'+\
				f'key={WEATHER_KEY}&q=Skopje&format=json'+\
				f'&date={curr_start.strftime("%Y-%m-%d")}'+\
				f'&enddate={curr_end.strftime("%Y-%m-%d")}'+\
				f'&tp=1'

			r = requests.get(url=URL) 
			data = r.json()

			for day in data['data']['weather']:
				day_date = TZ_MKD.localize(datetime.strptime(day['date'], "%Y-%m-%d"))
				for i, hour_data in enumerate(day['hourly']):
					results.append(((day_date + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"), hour_data['tempC'], hour_data['windspeedKmph'],
						np.sin((2*np.pi/360)*float(hour_data['winddirDegree'])), np.cos((2*np.pi/360)*float(hour_data['winddirDegree'])), 
						hour_data['precipMM'], hour_data['humidity'], hour_data['visibility'], hour_data['pressure'], 
						hour_data['cloudcover'], hour_data['DewPointC'], hour_data['uvIndex']))

			# reset penalty for waiting
			sleep_penalty_time = 5

			curr_start += period

		except Exception as e:
			print(e)
			time.sleep(sleep_penalty_time)
			# Increase penalty in case of an exception
			sleep_penalty_time += 5

			# If we keep throwing multiple consecutive exceptions, terminate
			if sleep_penalty_time > 30:
				raise Exception('Cannot fetch historic weather data')
		
	if not os.path.exists(f'./data/{pipeline_type}'):
		os.makedirs(f'./data/{pipeline_type}')

	# append the data to the file
	with open(f'./data/{pipeline_type}/weather', 'w') as f:
		f.write('Timestamp,temperature,wind_speed,wind_dir_sin,wind_dir_cos,precip,humidity,visibility,pressure,' + 
				'cloud_cover,dew_point,uv_index\n')
		for item in results:
			f.write(",".join([str(attribute) for attribute in item]) + "\n")


def fetch_future_weather(pipeline_type):
	"""
		This utility function fetches forecasted weather data for the next 5 days.
		
		Arguments:
			pipeline_type -- (this argument is always PIPELINE_INFERENCE_DECODER) 
	"""
	results = []
	try:
		print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\t{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")} - {(datetime.now(TZ_MKD) + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")}\tweather')

		URL = f'http://api.worldweatheronline.com/premium/v1/weather.ashx?key={WEATHER_KEY}&q=Skopje&format=json&num_of_days=5&tp=1'
		r = requests.get(url=URL) 
		data = r.json()

		for day in data['data']['weather']:
			day_date = TZ_MKD.localize(datetime.strptime(day['date'], "%Y-%m-%d"))
			for i, hour_data in enumerate(day['hourly']):
				results.append(((day_date + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"), hour_data['tempC'], hour_data['windspeedKmph'],
					np.sin((2*np.pi/360)*float(hour_data['winddirDegree'])), np.cos((2*np.pi/360)*float(hour_data['winddirDegree'])), 
					hour_data['precipMM'], hour_data['humidity'], hour_data['visibility'], hour_data['pressure'], 
					hour_data['cloudcover'], hour_data['DewPointC'], hour_data['uvIndex']))

	except Exception as e:
		print(e)
		raise Exception('Cannot fetch future weather data')

	if not os.path.exists(f'./data/{pipeline_type}'):
		os.makedirs(f'./data/{pipeline_type}')

	# append the data to the file
	with open(f'./data/{pipeline_type}/weather', 'w') as f:
		f.write('Timestamp,temperature,wind_speed,wind_dir_sin,wind_dir_cos,precip,humidity,visibility,pressure,' + 
				'cloud_cover,dew_point,uv_index\n')
		for item in results:
			f.write(",".join([str(attribute) for attribute in item]) + "\n")


def fetch_historical_data(date_start, date_end, pipeline_type):
	"""
		This utility function fetches historical data consisting of: air pollution, datetime features and 
		weather features. The joined results are stored in './data/pipeline_type/first-order', hence the 
		naming of this dataset as first-order dataset. This utility function is used for fetching data 
		used during training, as well as fetching data for inference (required by the encoder).
		
		Arguments:
			date_start -- beginning of fetching period
			date_end -- ending of fetching period
			pipeline_type -- type of data pipeline (possible values: PIPELINE_TRAINING, PIPELINE_INFERENCE_ENCODER)
	"""
	# Obtain the individual datasets
	fetch_air_pollution(date_start, date_end, pipeline_type)
	fetch_datetime(date_start, date_end, pipeline_type)
	fetch_historic_weather(date_start, date_end, pipeline_type)

	# Merge the datasets (first-order)
	df_weather = pd.read_csv(f'./data/{pipeline_type}/weather', index_col=0)
	df_datetime = pd.read_csv(f'./data/{pipeline_type}/datetime', index_col=0)

	if not os.path.exists(f'./data/{pipeline_type}/first-order'):
		os.makedirs(f'./data/{pipeline_type}/first-order')

	for station in SENSORS:
		df_sensor = pd.read_csv(f'./data/{pipeline_type}/formatted-sensor/{station}', index_col=0)
		df = df_sensor.join(df_datetime)
		df = df.join(df_weather)
		df.to_csv(f'./data/{pipeline_type}/first-order/{station}', index=True)


def fetch_future_data(date_start, date_end, pipeline_type): 
	"""
		This utility function fetches future data consisting of: air pollution (current hour), 
		datetime features (next 12 hours) and weather features (next 12 hours). The joined results 
		are stored in './data/pipeline_type/first-order', hence the naming of this dataset as first-order 
		dataset. This utility function is used for fetching data for inference (required by the decoder).
		
		Arguments:
			date_start -- beginning of fetching period
			date_end -- ending of fetching period
			pipeline_type -- type of data pipeline (this argument is always PIPELINE_INFERENCE_DECODER) 
	"""
	# Obtain the air pollution for the current hour
	fetch_air_pollution(date_start=TZ_MKD.localize(datetime(date_start.year, date_start.month, date_start.day, date_start.hour, 0, 0)), 
						date_end=date_start, pipeline_type=pipeline_type) 
	# obtain the datetime features for the next 12 hours
	fetch_datetime(date_start, date_end, pipeline_type)
	# obtain the weather features for the next 12 hours
	fetch_future_weather(pipeline_type)

	# Merge the datasets (first-order)
	df_weather = pd.read_csv(f'./data/{pipeline_type}/weather', index_col=0)
	df_datetime = pd.read_csv(f'./data/{pipeline_type}/datetime', index_col=0)

	if not os.path.exists(f'./data/{pipeline_type}/first-order'):
		os.makedirs(f'./data/{pipeline_type}/first-order')

	df = df_datetime.join(df_weather)
	df.to_csv(f'./data/{pipeline_type}/first-order/extras', index=True)


def scale_data(df, station):
	"""
		This utility function scales a dataset for a given station by using the pre-fitted scalers.
		
		Arguments:
			df -- dataset to be scaled
			station -- the station from which the dataset is derived

		Returns:
			df_scaled -- scaled dataset
	"""
	df_scaled = df.copy()
	for feature in os.listdir(f'./pickles/scalers/{station}'):
		with open(f'./pickles/scalers/{station}/{feature}', 'rb') as f:
			scaler = pickle.load(f)
			df_scaled[feature] = scaler.transform(df[feature].values.reshape(-1,1)).flatten()
	
	return df_scaled


def build_seq2seq_datasets(dataset, pollutants, history = 24, target_size = 12):
	"""
		This utility function builds datasets that are of appropriate dimensions for the seq2seq models.
		In particular, from each dataset, it builds three separate datasets: encoder_input_data of shape
		(?, history, num_encoder_features), decoder_input_data of shape (?, target_size, num_decoder_features) 
		and decoder_target_data of shape (?, target_size, num_pollutants). These datasets are used for 
		training the seq2seq models.
		
		Arguments:
			dataset -- dataset to be transformed/reshaped.
			pollutants -- list of pollutants in the dataset
			history -- lookback period for the models (len of input sequence)
			target_size - period to be predicted by the models (len of output sequence)

	"""
	start_index = history
	end_index = len(dataset) - target_size
	
	# Selecting the appropriate columns from the dataset
	encoder_input_dataset = dataset.values.copy()
	decoder_input_dataset = dataset.drop([c for c in dataset.columns if 'missing' in c], 
										 axis=1).values.copy()
	decoder_target_dataset = dataset[pollutants].values.copy()
	decoder_missing_dataset = dataset[[f'{p}_missing' for p in pollutants]].values.copy()

	# These lists will hold the final (third-order) datasets
	encoder_input_data = []
	decoder_input_data = []
	decoder_target_data = []
	
	for i in range(start_index, end_index):
		encoder_input_values = encoder_input_dataset[i-history:i]
		decoder_input_values = decoder_input_dataset[i:i+target_size]
		decoder_target_values = decoder_target_dataset[i+1:i+1+target_size]
		decoder_missing_values = decoder_missing_dataset[i+1:i+1+target_size]
		
		# If any of the target values has been imputed (i.e. was missing), skip the sample
		if np.any(decoder_missing_values == 1):
			continue
			
		encoder_input_data.append(encoder_input_values)
		decoder_input_data.append(decoder_input_values)
		decoder_target_data.append(decoder_target_values)

	encoder_input_data = np.array(encoder_input_data).reshape(-1, 
															  history, 
															  encoder_input_dataset.shape[1])
	decoder_input_data = np.array(decoder_input_data).reshape(-1, 
															  target_size, 
															  decoder_input_dataset.shape[1])
	decoder_target_data = np.array(decoder_target_data).reshape(-1, 
																target_size, 
																decoder_target_dataset.shape[1])
		
	return encoder_input_data, decoder_input_data, decoder_target_data


def training_data_pipeline():
	"""
		This function fetches and transforms the data needed for training the models. 
	"""
	# Fetch data
	print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tFetching historical data for training')
	fetch_historical_data(date_start=TZ_MKD.localize(datetime(2011, 1, 1, 0, 0, 0)), date_end=datetime.now(TZ_MKD), 
						  pipeline_type=PIPELINE_TRAINING)

	# Transform data
	for station, pollutants in SENSORS.items(): 
		print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tProcessing training data for {station}')
		df = pd.read_csv(f'./data/training/first-order/{station}', index_col=0)

		# add feature indicating missingness for each pollutant
		for p in pollutants:
			df[f'{p}_missing'] = df[f'{p}'].isna().astype('int32')

		# log-transform pollutants
		for p in pollutants:
			df[p] = np.log(df[p] + 1)

		# train-val split
		train_size = int(df.shape[0] * 0.85)
		df_train = df.iloc[:train_size]
		df_valid = df.iloc[train_size:]

		# fit and save scalers
		features_to_normalize = ['cloud_cover', 'precip', 'uv_index', 'visibility']
		features_to_standardize =  pollutants + ['temperature', 'humidity', 'dew_point',
												'pressure', 'wind_speed']
		scalers = {}
		for f in features_to_normalize:
			scaler = MinMaxScaler()
			scaler.fit(df_train[f].values.reshape(-1,1))
			scalers[f] = scaler

		for f in features_to_standardize:
			scaler = StandardScaler()
			scaler.fit(df_train[f].values.reshape(-1,1))
			scalers[f] = scaler

		if not os.path.exists(f'./pickles/scalers/{station}'):
			os.makedirs(f'./pickles/scalers/{station}')

		for feature, scaler in scalers.items():
			with open(f'./pickles/scalers/{station}/{feature}', 'wb') as f:
				pickle.dump(scaler, f)

		df_train_scaled = scale_data(df_train, station)
		df_valid_scaled = scale_data(df_valid, station)
		train_values = df_train_scaled.values.copy()
		valid_values = df_valid_scaled.copy()

		# impute missing values
		imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=12, random_state=0), 
								   random_state=0, skip_complete=True, max_iter=5)

		imputed_train_values = imputer.fit_transform(train_values)
		imputed_valid_values = imputer.transform(valid_values)

		if not os.path.exists(f'./pickles/imputers'):
			os.makedirs(f'./pickles/imputers')

		with open(f'./pickles/imputers/{station}', 'wb') as f:
			pickle.dump(imputer, f)

		df_train_imputed = pd.DataFrame(data=imputed_train_values, 
								index=df_train_scaled.index,
								columns=df_train_scaled.columns)
		
		df_valid_imputed = pd.DataFrame(data=imputed_valid_values, 
										index=df_valid_scaled.index,
										columns=df_valid_scaled.columns)

		if not os.path.exists(f'./data/training/second-order/{station}'):
			os.makedirs(f'./data/training/second-order/{station}')

		df_train_imputed.to_csv(f'./data/training/second-order/{station}/train', index=True)
		df_valid_imputed.to_csv(f'./data/training/second-order/{station}/valid', index=True)

		# build seq2seq (third-order) datasets
		train_encoder_input_data, train_decoder_input_data, train_decoder_target_data = \
			build_seq2seq_datasets(df_train_imputed, pollutants)

		valid_encoder_input_data, valid_decoder_input_data, valid_decoder_target_data = \
			build_seq2seq_datasets(df_valid_imputed, pollutants)

		if not os.path.exists(f'./data/training/third-order/{station}'):
			os.makedirs(f'./data/training/third-order/{station}')

		np.save(f'./data/training/third-order/{station}/train_encoder_input_data.npy', train_encoder_input_data)
		np.save(f'./data/training/third-order/{station}/train_decoder_input_data.npy', train_decoder_input_data)
		np.save(f'./data/training/third-order/{station}/train_decoder_target_data.npy', train_decoder_target_data)

		np.save(f'./data/training/third-order/{station}/valid_encoder_input_data.npy', valid_encoder_input_data)
		np.save(f'./data/training/third-order/{station}/valid_decoder_input_data.npy', valid_decoder_input_data)
		np.save(f'./data/training/third-order/{station}/valid_decoder_target_data.npy', valid_decoder_target_data)


def inference_data_pipeline():
	"""
		This function fetches and transforms the data needed by the models during inference time. 
	"""
	# Fetch historical data
	print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tFetching encoder input data for inference')
	now = datetime.now(TZ_MKD)
	date_history_end = now - timedelta(hours=1)
	date_history_start = now - timedelta(hours=24)
	fetch_historical_data(date_start=date_history_start, date_end=date_history_end,
						pipeline_type=PIPELINE_INFERENCE_ENCODER)

	# Fetch future data
	print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tFetching decoder input data for inference')
	date_future_start = now
	date_future_end = now + timedelta(hours=11)
	fetch_future_data(date_start=date_future_start, date_end=date_future_end,
					pipeline_type=PIPELINE_INFERENCE_DECODER)

	# Transform historical data
	for station, pollutants in SENSORS.items(): 
		print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tProcessing encoder input data for {station}')
		df = pd.read_csv(f'./data/inference-encoder/first-order/{station}', index_col=0)

		# add feature indicating missingness for each pollutant
		for p in pollutants:
			df[f'{p}_missing'] = df[f'{p}'].isna().astype('int32')

		# log-transform pollutants
		for p in pollutants:
			df[p] = np.log(df[p] + 1)

		# scale data
		df_scaled = scale_data(df, station)

		# impute data
		with open(f'./pickles/imputers/{station}', 'rb') as f:
			imputer = pickle.load(f)
		imputed = imputer.transform(df_scaled)

		# reshape data
		encoder_input_data = np.expand_dims(imputed, axis=0)

		if not os.path.exists(f'./data/inference-encoder/third-order/{station}'):
			os.makedirs(f'./data/inference-encoder/third-order/{station}')

		np.save(f'./data/inference-encoder/third-order/{station}/encoder_input_data.npy', encoder_input_data)

	# Transform future data
	for station, pollutants in SENSORS.items(): 
		print(f'[{datetime.now(TZ_MKD).strftime("%Y-%m-%d %H:%M:%S")}]\tProcessing decoder input data for {station}')

		# load extras dataset
		df_extras = pd.read_csv(f'./data/inference-decoder/first-order/extras', index_col=0)

		# scale the weather features
		for feature in os.listdir(f'./pickles/scalers/{station}'):
			if feature not in pollutants:
				with open(f'./pickles/scalers/{station}/{feature}', 'rb') as f:
					scaler = pickle.load(f)
					df_extras[feature] = scaler.transform(df_extras[feature].values.reshape(-1,1)).flatten()

		# reshape extras data
		decoder_input_extras_data = np.expand_dims(df_extras.values, axis=0)

		if not os.path.exists(f'./data/inference-decoder/third-order/{station}'):
			os.makedirs(f'./data/inference-decoder/third-order/{station}')

		# save extras data
		np.save(f'./data/inference-decoder/third-order/{station}/decoder_input_extras_data.npy', decoder_input_extras_data)

		# load pollution dataset
		df_pollution = pd.read_csv(f'./data/inference-decoder/formatted-sensor/{station}', index_col=0)

		# log-transform and scale pollutants
		for p in pollutants:
			df_pollution[p] = np.log(df_pollution[p] + 1)
			with open(f'./pickles/scalers/{station}/{p}', 'rb') as f:
				scaler = pickle.load(f)
				df_pollution[p] = scaler.transform(df_pollution[p].values.reshape(-1,1)).flatten()

		pollutants_now = df_pollution.values
		extras_now = decoder_input_extras_data[0, 0, :].reshape(1, -1)
		missing_now = np.zeros((1, len(pollutants)))
		full_now = np.concatenate([pollutants_now, extras_now, missing_now], axis=-1)

		# impute pollutants
		with open(f'./pickles/imputers/{station}', 'rb') as f:
			imputer = pickle.load(f)
		imputed = imputer.transform(full_now)

		decoder_input_pollution_data = imputed[:, :len(pollutants)]

		# save pollution data
		np.save(f'./data/inference-decoder/third-order/{station}/decoder_input_pollution_data.npy', decoder_input_pollution_data)
