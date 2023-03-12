import numpy as np
import pandas as pd
import csv
import ctypes
# import flask.scaffold

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from owlready2 import *


# from pandas import Timestamp
# from flask import Flask, jsonify, Blueprint, render_template
# from prometheus_flask_exporter import PrometheusMetrics


# app = Flask(__name__)
# metricsProm = PrometheusMetrics(app)
# blueprint = Blueprint('api', __name__, url_prefix='/api')
# api = Api(blueprint, doc='/documentation', title = "Search anomaly",
#		  description = "Search anomaly") #,doc=False
# api.namespace('names', description='Manage names')

# app.register_blueprint(blueprint)

# app.config['SWAGGER_UI_JSONEDITOR'] = True

def SearchAnomaly(df, axex):
    split = 0.5

    cutoff = int(len(df) * split)
    train_df = df.head(cutoff)
    test_df = df.tail(len(df) - cutoff)

    training_mean = train_df.mean()
    training_std = train_df.std()
    df_training_value = (train_df - training_mean) / training_std

    df_test_value = (test_df - training_mean) / training_std
    # fig, ax = plt.subplots()
    # df_test_value.plot(legend=False, ax=ax)
    # plt.show()

    # print(df_training_value)

    TIME_STEPS = 12

    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(0, len(values) - time_steps + 1):
            output.append(values[i: (i + time_steps)])
            # print(i, values[(i - time_steps//2): (i + time_steps//2)])
            # pd.DataFrame(array).plot(ax=axex, label=str(i),color="b")
            # pd.DataFrame(x_train[0][i], index=range(0, len(x_train[0][i]))).plot(ax=axex, label = str(i), color="black")

        return np.stack(output)

    # print(df_training_value.values)
    # print(df_test_value.values)

    #print(df_training_value.values)
    x_train = create_sequences(df_training_value.values)
    #print("Training input shape: ", x_train.shape)
    print(x_train)

    x_test = create_sequences(df_test_value.values)
    #print("Test input shape: ", x_test.shape)
    #print(x_test)

    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),

            # layers.LSTM(32, return_sequences=True, activation="relu"),

            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same", batch_size=50),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=50,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    x_train_pred = model.predict(x_train)

    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    threshold = np.max(train_mae_loss)

    x_test_pred = model.predict(x_test)
    x_test_pred = x_test_pred.transpose()

    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    # test_mae_loss = test_mae_loss.reshape((-1))

    anomalies = test_mae_loss > threshold
    # print(anomalies)
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):

        # print(np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]))
        if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
            anomalous_data_indices.append(data_idx)
    df_subset = test_df.iloc[anomalous_data_indices]
    # print(df_subset)

    return df_subset


def UsingOntology(df_anomaly):
    onto = get_ontology("prototype_rasshir.owl").load()

    for individual in onto.individuals():
        destroy_entity(individual)
    # print(df_anomaly)
    #onto.save(file="prototypePython.owl", format="rdfxml")
    with onto:

        # class Well2(Thing):
        #     namespace = onto
        #     pass
        #
        # class hasWell_Name2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [str]
        #
        # class hasDate2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [str]
        #
        # class hasDeltaPHI2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [float]
        #
        # class hasDepth2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [float]
        #
        # class hasGR2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [float]

        well = None
        index = 0
        listMesure = []
        for date, anomaly in df_anomaly.iterrows():
            # print(str(index))
            if well == None or well.name != anomaly['Well Name']:
                well = onto.Well(anomaly['Well Name'])

            DeltaPHI = onto.DeltaPHI("DeltaPHI" + str(index),
                                     hasDeltaPHI=anomaly['DeltaPHI'])
            Depth = onto.Depth("Depth" + str(index),
                               hasDepth=anomaly['Depth'])
            Gamma_emission = onto.Gamma_emission("Gamma_emission" + str(index),
                                                 hasGR=anomaly['GR'])
            Index_petrophysics = onto.Index_petrophysics("Index_petrophysics" + str(index),
                                                         hasIndexP=anomaly['ILD_log10'])
            measurment = onto.Measurement("Measurement" + str(index),
                                          hasDate=str(date))

            list = []
            list.append(DeltaPHI)
            list.append(Depth)
            list.append(Gamma_emission)
            list.append(Index_petrophysics)
            measurment.hasMetrics = list
            listMesure.append(measurment)

            index += 1
        if well is not None:
            well.hasMeasurement = listMesure


        # sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=3)

        #onto.save(file="prototypePython.owl", format="rdfxml")

    ctypes.windll.user32.MessageBoxW(0, "Запустите правила", "Пауза", 1)

    onto2 = get_ontology("prototypePython.owl").load()
    # print(df_anomaly.index)
    anomalous_data_indices = []
    for SWRLanomaly in onto2.search(is_a = onto2.Measurement):
        print(SWRLanomaly.hasAnomaly)
        if SWRLanomaly.hasAnomaly is not None and SWRLanomaly.hasDate is not None:
            # print(pd.to_datetime(SWRLanomaly.hasDate, format='%Y-%m-%d') unit='s'))

            anomalous_data_indices.append([pd.to_datetime(SWRLanomaly.hasDate, format='%Y-%m-%d'), SWRLanomaly.hasAnomaly])
    print(anomalous_data_indices)
    # onto.save(file="prototypePython.owl", format="rdfxml")

    return anomalous_data_indices


def printDF(df, axex):
    # plt.ion()

    dfNames = pd.DataFrame(pd.unique(df['Well Name']))

    dfNames = dfNames.drop(np.where(dfNames[0] == 'Recruit F9')[0]).to_numpy()
    # print(dfNames)
    # ['CHURCHMAN BIBLE' 'CROSS H CATTLE' 'LUKE G U' 'NEWBY' 'NOLAN' 'Recruit F9' 'SHANKLE' 'SHRIMPLIN']

    i = 1
    for Name in dfNames[3:4]:
        Name = Name[0]
        dataForName = df.loc[df['Well Name'] == Name]
        dataForName = dataForName.sort_values("Depth")


        # axex.plot(dataForName.index, dataForName["PHIND"], label="PHIND")

        # axex.plot(dataForName.index, dataForName["GR"], label="GR")

        # axex.plot(dataForName.index, dataForName["ILD_log10"], label="ILD_log10")

        # axex.plot(dataForName.index, dataForName["DeltaPHI"], label="DeltaPHI")
        # axex.legend()

        # plt.show()

        # plt.plot(data["date"], data["DeltaPHI"], label=Name)
        # plt.figure(i, figsize=(5, 5))

        # plt.subplot(1, 9, i)

        # df_subset = SearchAnomaly(dataForName[["GR"]], axex)
        df_subset = []
        # print(df_subset)
        # plt.subplots_adjust(wspace=0, hspace=i)
        dataForName[["GR"]].plot(ax=axex[0], legend=True, color="black")
        dataForName[["DeltaPHI"]].plot(ax=axex[1], legend=True, color="black")
        dataForName[["ILD_log10"]].plot(ax=axex[2], legend=True, color="black")
        # df_subset = dataForName.loc[df_subset.index]
        #if len(df_subset) > 0:
            #df_subset[["GR"]].plot(ax=axex[0], legend=False, color="r", marker='s', linewidth=0)
            #df_subset[["DeltaPHI"]].plot(ax=axex[1], legend=False, color="r", marker='s', linewidth=0)
            #df_subset[["ILD_log10"]].plot(ax=axex[2], legend=False, color="r", marker='s', linewidth=0)

        #owl_ontology = UsingOntology(df_subset)
        # print("anomaly")
        # print(owl_ontology)
        owl_ontology = []
        for anomaly in owl_ontology:

            if anomaly[0] is None:
                continue
            df_subset = pd.DataFrame(dataForName.loc[anomaly[0]]).transpose()
            print(df_subset)
            #print(dataForName)
            color = None
            print(anomaly[1][0])
            if anomaly[1][0] == "rule1":
                color = "b"
            elif anomaly[1][0] == "rule2":
                color = "y"
            elif anomaly[1][0] == "rule3":
                color = "pink"

            df_subset["GR"].plot(ax=axex[0], legend=False, color=color, marker='o', linewidth=0)
            #axex[0].plot(y=df_subset["GR"], x=, legend=False, color=color, marker='o', linewidth=0)
            df_subset["DeltaPHI"].plot(ax=axex[1], legend=False, color=color, marker='o', linewidth=0)
            #df_subset["DeltaPHI"].plot(ax=axex[1], legend=False, color=color, marker='o', linewidth=0)
            df_subset["ILD_log10"].plot(ax=axex[2], legend=False, color=color, marker='o', linewidth=0)
            #df_subset["ILD_log10"].plot(ax=axex[2], legend=False, color=color, marker='o', linewidth=0)

        i += 1
    # plt.ioff()
    #plt.show()
    return axex



# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.expand_frame_repr = False

def startSearch(axex):

    with open('../Sourses/facies_data.csv', newline='') as csvfile:
    # with open(path, newline='') as csvfile:
        date = datetime.date.today()
        dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=',', quotechar='|'))

    # print(df.columns)
    # ['Facies','Formation','Well Name','Depth','GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS','date']

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

    # print(dfSource)
    axex = printDF(dfSource, axex)

    return axex

# @app.route('/')
# def index():
#    return render_template("index.html",tables=[dfSource.to_html()], titles=dfSource.columns.values)
# return render_template("data.html",data=dfSource.to_html())

# if __name__ == "__main__":
#    app.run()
