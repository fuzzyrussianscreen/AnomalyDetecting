import csv
import ctypes

import numpy as np
import pandas as pd
from owlready2 import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.expand_frame_repr = False


def SearchAnomaly(df_full, df):
	split = 0.5
	# split = 200/len(df)

	cutoff = int(len(df) * split)
	scaler = MinMaxScaler(feature_range=(0, 1))

	scaleddf = df.loc[:, ['GR']]
	scaleddf["GR"] = scaler.fit_transform(df.loc[:, ["GR"]].values.reshape(-1, 1))


	train_df = scaleddf.head(cutoff)
	test_df = scaleddf.tail(len(scaleddf) - cutoff)
	train_df_for_exper = scaleddf.head(int(len(df)*0.5))


	# training_mean = train_df.mean()
	# training_std = train_df.std()
	# df_training_value = (train_df - training_mean) / training_std

	# df_test_value = (test_df - training_mean) / training_std

	TIME_STEPS = 12

	def create_sequencesXY(values, time_steps=TIME_STEPS):
		x, y = [], []
		# for i in range(0, len(values) - time_steps + 1):
		for i in range(time_steps, len(values)):
			x.append(values[i-time_steps:i, 0])
			y.append(values[i, 0])

		return np.array(x), np.array(y)

	def	create_sequencesX(values, time_steps=TIME_STEPS):
		x = []
		for i in range(time_steps, len(values)):
			x.append(values[i - time_steps:i, 0])

		return np.array(x)

	x_train, y_train = create_sequencesXY(train_df.values)
	x_train_for_exper, y_train_for_exper = create_sequencesXY(train_df_for_exper.values)
	x_test = create_sequencesX(test_df.values)

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_train_for_exper = np.reshape(x_train_for_exper, (x_train_for_exper.shape[0], x_train_for_exper.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	# print(x_train.shape)
	# print(y_train.shape)
	# print(x_test.shape)

	kernel_size = 7
	strides= 2
	filters = 32
	activation = "selu"


	# model = keras.Sequential(
	# 	[
	# 		layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
	# 		layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", strides=strides, activation=activation),
	# 		layers.Dropout(rate=0.2),
	# 		layers.Conv1D(filters=filters/2, kernel_size=kernel_size, padding="same", strides=strides, activation=activation),
	# 		layers.Conv1DTranspose(filters=filters/2, kernel_size=kernel_size, padding="same", strides=strides, activation=activation),
	# 		layers.Dropout(rate=0.2),
	# 		layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding="same", strides=strides, activation=activation),
	# 		layers.Conv1DTranspose(filters=1, kernel_size=kernel_size, padding="same", batch_size=50),
	# 		layers.Dense(1, activation='sigmoid', name='decoder_dense')
	# 	]
	# )



	model = keras.Sequential([
		layers.LSTM(50, activation='softsign', input_shape=(x_train.shape[1], x_train.shape[2])),
		layers.Dense(1, activation='sigmoid', name='decoder_dense')])

	model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001), loss="mse", metrics=[
																									# metrics.Accuracy(),
																									# metrics.Precision(),
																									# metrics.AUC(),
																									metrics.MeanAbsoluteError(),
																									# metrics.MeanSquaredError(),
																									# metrics.Recall()
																									])
	model.summary()

	history = model.fit(x_train, y_train, epochs=300, batch_size=50, validation_split=0.1, verbose=0,
	                    callbacks=[keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=10, mode="min")])

	# print(pd.DataFrame(history.history))

	mse = model.evaluate(x_train_for_exper, y_train_for_exper, verbose=0)

	print(model.metrics_names)
	print(mse)

	x_train_pred = model.predict(x_train)
	train_mae_loss = np.mean(np.abs(x_train_pred - y_train), axis=1)
	threshold = np.max(train_mae_loss)

	x_test_pred = model.predict(x_test)
	x_test_pred = x_test_pred.transpose()

	test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

	# print(x_test_pred)
	# print(x_test)

	# prec = metrics.Precision()
	# prec.update_state(x_test_pred, x_test)

	# print("Precision "+prec.result().numpy())

	anomalies = test_mae_loss > threshold
	anomalous_data_indices = []
	for data_idx in range(TIME_STEPS - 1, len(x_test) - TIME_STEPS + 1):
		if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
			anomalous_data_indices.append(data_idx)
	df_subset = df_full.tail(len(df_full) - cutoff).iloc[anomalous_data_indices]

	return df_subset

def LoadSWRL(path):
	onto = get_ontology("../../Sources/ontology/ontology_full.owl").load()
	with onto:
		return onto.rules()

def SaveSWRL(path, new_rules):
	onto = get_ontology("../../Sources/ontology/ontology_empty.owl").load()
	with onto:

		swrl_assertions = list(onto.rules())
		for swrl_assertion in swrl_assertions:
			destroy_entity(swrl_assertion)
			#onto.world.remove(swrl_assertion)

		for new_rule_text in new_rules:
			print(new_rule_text)
			new_rule = Imp()
			new_rule.set_as_rule(new_rule_text)
		onto.save(file="../../Sources/ontology/ontology_empty.owl", format="rdfxml")


def UsingOntology(df_anomaly, path):
	ontoCopy = get_ontology(path).load()

	with ontoCopy:
		for individual in ontoCopy.individuals():
			destroy_entity(individual)
		ontoCopy.save(file="../../Sources/ontology/ontology_full.owl", format="rdfxml")
	onto = get_ontology("../../Sources/ontology/ontology_full.owl").load()
	with onto:

		well = None
		index = 0
		listMesure = []
		#print(len(df_anomaly))
		if len(df_anomaly) > 1:

			for date, anomaly in df_anomaly.iterrows():
				# print(str(index))
				if well == None or well.name != anomaly['Well Name']:
					well = onto.Well(anomaly['Well Name'])

				DeltaPHI = onto.DeltaPHI("DeltaPHI" + str(index), hasDeltaPHI=anomaly['DeltaPHI'])
				Depth = onto.Depth("Depth" + str(index), hasDepth=anomaly['Depth'])
				Gamma_emission = onto.Gamma_emission("Gamma_emission" + str(index), hasGR=anomaly['GR'])
				Index_petrophysics = onto.Index_petrophysics("Index_petrophysics" + str(index),
				                                             hasIndexP=anomaly['ILD_log10'])
				measurment = onto.Measurement("Measurement" + str(index), hasDate=str(date))

				list = [DeltaPHI, Depth, Gamma_emission, Index_petrophysics]
				measurment.hasMetrics = list
				listMesure.append(measurment)

				index += 1
			if well is not None:
				well.hasMeasurement = listMesure

			# sync_reasoner_hermit(infer_property_values=True, debug=3)
			sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
			onto.save(file="../../Sources/ontology/ontology_full.owl", format="rdfxml")

	#ctypes.windll.user32.MessageBoxW(0, "Запустите правила", "Пауза", 1)



	onto = get_ontology("../../Sources/ontology/ontology_full.owl").load()
	with onto:
		# print(df_anomaly.index)
		anomalous_data_indices = []
		for SWRLanomaly in onto.search(is_a=onto.Measurement):
			print(SWRLanomaly.hasAnomaly)
			if SWRLanomaly.hasAnomaly != [] and SWRLanomaly.hasDate is not None:
				# print(pd.to_datetime(SWRLanomaly.hasDate, format='%Y-%m-%d'))

				anomalous_data_indices.append(
					[pd.to_datetime(SWRLanomaly.hasDate, format='%Y-%m-%d'), SWRLanomaly.hasAnomaly])
	# print(anomalous_data_indices)
	# onto2.save(file="../Sourses/ontology_full.owl", format="rdfxml")

	#print(anomalous_data_indices)
	return anomalous_data_indices


def printDFOntology(axex, dataForName, df_subset, owl_ontology):
	#print(dataForName)
	#print(df_subset)
	#print(owl_ontology)

	dataForName[["GR"]].plot(ax=axex[0], legend=False, color="black")
	dataForName[["DeltaPHI"]].plot(ax=axex[1], legend=False, color="black")
	dataForName[["ILD_log10"]].plot(ax=axex[2], legend=False, color="black")
	df_subset = dataForName.loc[df_subset.index]

	if len(df_subset) > 0:
		df_subset[["GR"]].plot(ax=axex[0], legend=False, color="r", marker='s', linewidth=0)
		df_subset[["DeltaPHI"]].plot(ax=axex[1], legend=False, color="r", marker='s', linewidth=0)
		df_subset[["ILD_log10"]].plot(ax=axex[2], legend=False, color="r", marker='s', linewidth=0)

	for anomaly in owl_ontology:

		if anomaly[0] is None:
			continue
		df_subset = pd.DataFrame(dataForName.loc[anomaly[0]]).transpose()
		color = None
		# print(anomaly[1][0] )
		if anomaly[1][0] == "rule1":
			color = "b"
		elif anomaly[1][0] == "rule2":
			color = "y"
		elif anomaly[1][0] == "rule3":
			color = "pink"
		else:
			color = "r"
		df_subset["GR"].plot(ax=axex[0], legend=False, color=color, marker='o', linewidth=0)
		df_subset["DeltaPHI"].plot(ax=axex[1], legend=False, color=color, marker='o', linewidth=0)
		df_subset["ILD_log10"].plot(ax=axex[2], legend=False, color=color, marker='o', linewidth=0)

	return axex


def printDFNeuron(axex, dataForName, df_subset):
	dataForName[["GR"]].plot(ax=axex[0], legend=True, color="black")
	dataForName[["DeltaPHI"]].plot(ax=axex[1], legend=True, color="black")
	dataForName[["ILD_log10"]].plot(ax=axex[2], legend=True, color="black")

	df_subset = dataForName.loc[df_subset.index]
	if len(df_subset) > 0:
		df_subset[["GR"]].plot(ax=axex[0], legend=True, label='anomaly measurements', color="r", marker='s',
		                       linewidth=0)
		df_subset[["DeltaPHI"]].plot(ax=axex[1], legend=True, label='anomaly measurements', color="r", marker='s',
		                             linewidth=0)
		df_subset[["ILD_log10"]].plot(ax=axex[2], legend=True, label='anomaly measurements', color="r", marker='s',
		                              linewidth=0)

	return axex


def printDF(axex, dataForName):
	dataForName[["GR"]].plot(ax=axex[0], legend=True, color="black")
	dataForName[["DeltaPHI"]].plot(ax=axex[1], legend=True, color="black")
	dataForName[["ILD_log10"]].plot(ax=axex[2], legend=True, color="black")

	return axex


def startSearch(path):
	with open(path, newline='') as csvfile:
		date = datetime.date.today()
		dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=';', quotechar='|'))

		dfSource.columns = dfSource.iloc[0]
		dfSource.drop([0], inplace=True)
		dfSource = dfSource.reset_index(drop=True)
		dfSource["date"] = pd.date_range(start=date, end=date + datetime.timedelta(days=len(dfSource) - 1), freq='D')
		dfSource['date'] = pd.to_datetime(dfSource['date'], format='%Y-%m-%d %H:%M:%S')
		dfSource = dfSource.set_index('date')

		dfSource['Depth'] = dfSource['Depth'].astype(float)
		dfSource['DeltaPHI'] = dfSource['DeltaPHI'].astype(float)
		dfSource['ILD_log10'] = dfSource['ILD_log10'].astype(float)
		dfSource['GR'] = dfSource['GR'].astype(float)
		dfSource['Well Name'] = dfSource['Well Name'].astype(str)

		dfNames = pd.DataFrame(pd.unique(dfSource['Well Name']))
		# ['CHURCHMAN BIBLE' 'CROSS H CATTLE' 'LUKE G U' 'NEWBY' 'NOLAN' 'Recruit F9' 'SHANKLE' 'SHRIMPLIN']
		dfNames = dfNames.drop(np.where(dfNames[0] == 'Recruit F9')[0]).to_numpy()

		#for Name in dfNames[0]:
		#	Name = Name[0]
		#	dataForName = dfSource.loc[dfSource['Well Name'] == Name]

	# axex = printDF(dfSource, axex)

	return dfSource
