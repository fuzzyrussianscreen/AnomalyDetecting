import csv
import ctypes

import numpy as np
import pandas as pd
from owlready2 import *
from tensorflow import keras
from tensorflow.keras import layers

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.expand_frame_repr = False


def SearchAnomaly(df_full, df):
	split = 0.5
	# print(df)
	cutoff = int(len(df) * split)
	train_df = df.head(cutoff)
	test_df = df.tail(len(df) - cutoff)

	training_mean = train_df.mean()
	training_std = train_df.std()
	df_training_value = (train_df - training_mean) / training_std

	df_test_value = (test_df - training_mean) / training_std

	TIME_STEPS = 12

	def create_sequences(values, time_steps=TIME_STEPS):
		output = []
		for i in range(0, len(values) - time_steps + 1):
			output.append(values[i: (i + time_steps)])

		return np.stack(output)

	x_train = create_sequences(df_training_value.values)
	x_test = create_sequences(df_test_value.values)

	model = keras.Sequential(
		[
			layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
			layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
			layers.Dropout(rate=0.2),
			layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
			layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
			layers.Dropout(rate=0.2),
			layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
			layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same", batch_size=50),
		]
	)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
	model.summary()

	history = model.fit(x_train, x_train, epochs=50, batch_size=50, validation_split=0.1, verbose=0,
	                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")])

	x_train_pred = model.predict(x_train)
	train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
	threshold = np.max(train_mae_loss)

	x_test_pred = model.predict(x_test)
	x_test_pred = x_test_pred.transpose()

	test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

	anomalies = test_mae_loss > threshold
	anomalous_data_indices = []
	for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
		if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
			anomalous_data_indices.append(data_idx)
	df_subset = df_full.tail(len(df_full) - cutoff).iloc[anomalous_data_indices]

	return df_subset


def UsingOntology(df_anomaly, path):
	ontoCopy = get_ontology(path).load()

	with ontoCopy:
		for individual in ontoCopy.individuals():
			destroy_entity(individual)
		ontoCopy.save(file="../../Sources/ontology/ontology_full.owl", format="rdfxml")
	onto = get_ontology("../Sources/ontology/ontology_full.owl").load()
	with onto:

		well = None
		index = 0
		listMesure = []
		print(len(df_anomaly))
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
			onto.save(file="../../Sources/ontology/ontology_full.owl", format="rdfxml")

	ctypes.windll.user32.MessageBoxW(0, "Запустите правила", "Пауза", 1)

	onto = get_ontology("../Sources/ontology/ontology_full.owl").load()
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

	print(anomalous_data_indices)
	return anomalous_data_indices


def printDFOntology(axex, dataForName, df_subset, owl_ontology):
	print(dataForName)
	print(df_subset)
	print(owl_ontology)

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
		dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=',', quotechar='|'))

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

		for Name in dfNames[6:7]:
			Name = Name[0]
			dataForName = dfSource.loc[dfSource['Well Name'] == Name]

	# axex = printDF(dfSource, axex)

	return dataForName
