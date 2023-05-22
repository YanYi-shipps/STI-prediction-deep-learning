#LSTM loss function
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import set_random_seed
import math
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(2019)
set_random_seed(2019)
random.seed(2009)
os.environ['PYTHONHASHSEED'] = '0'

U11_dataset = pd.read_csv('./U11.SI weekly.csv', header = 0)
C38U_dataset = pd.read_csv('./C38U.SI weekly.csv', header = 0)
E5H_dataset = pd.read_csv('./E5H.SI weekly.csv', header = 0)

#Include significant parameters from correlation analysis in excel
USDSG = pd.read_csv('./USDSG.csv', header = 0)
E5H_dataset.loc[:,'USDSG_Price'] = USDSG.loc[:, 'Price'].values
E5H_dataset.loc[:,'USDSG_Open'] = USDSG.loc[:, 'Open'].values
E5H_dataset.loc[:,'USDSG_High'] = USDSG.loc[:, 'High'].values
E5H_dataset.loc[:,'USDSG_Low'] = USDSG.loc[:, 'Low'].values

GDP = pd.read_csv('./GDP.csv', header = 0)
E5H_dataset.loc[:,'GDP'] = GDP.loc[:, 'GDP'].values
E5H_dataset.loc[:,'InterestRate(real)'] = GDP.loc[:, 'InterestRate(real)'].values

SIBOR = pd.read_csv('./SIBOR.csv', header = None)
U11_dataset.loc[:,'SIBOR'] = SIBOR.loc[:,3].values
C38U_dataset.loc[:,'SIBOR'] = SIBOR.loc[:,3].values
E5H_dataset.loc[:,'SIBOR'] = SIBOR.loc[:,3].values

STI = pd.read_csv('./STI.csv', header = 0)
U11_dataset.loc[:,'STI_Open'] = STI.loc[:, 'Open'].values
U11_dataset.loc[:,'STI_High'] = STI.loc[:, 'High'].values
U11_dataset.loc[:,'STI_Low'] = STI.loc[:, 'Low'].values
U11_dataset.loc[:,'STI_Close'] = STI.loc[:, 'Close'].values
U11_dataset.loc[:,'STI_Volume'] = STI.loc[:, 'Volume'].values

DJIA = pd.read_csv('./DJIA.csv', header = 0)
U11_dataset.loc[:,'DJIA_Open'] = DJIA.loc[:, 'Open'].values
U11_dataset.loc[:,'DJIA_High'] = DJIA.loc[:, 'High'].values
U11_dataset.loc[:,'DJIA_Low'] = DJIA.loc[:, 'Low'].values
U11_dataset.loc[:,'DJIA_Close'] = DJIA.loc[:, 'Close'].values
U11_dataset.loc[:,'DJIA_Volume'] = DJIA.loc[:, 'Volume'].values
C38U_dataset.loc[:,'DJIA_Open'] = DJIA.loc[:, 'Open'].values
C38U_dataset.loc[:,'DJIA_High'] = DJIA.loc[:, 'High'].values
C38U_dataset.loc[:,'DJIA_Low'] = DJIA.loc[:, 'Low'].values
C38U_dataset.loc[:,'DJIA_Close'] = DJIA.loc[:, 'Close'].values
C38U_dataset.loc[:,'DJIA_Volume'] = DJIA.loc[:, 'Volume'].values

#Normalise data
U11_dataset = U11_dataset.values
C38U_dataset = C38U_dataset.values
E5H_dataset = E5H_dataset.values

U11_date = U11_dataset[: , 0] #first column is date, save it first 
C38U_date = C38U_dataset[: , 0] #first column is date, save it first
E5H_date = E5H_dataset[: , 0] #first column is date, save it first

U11_scaler = MinMaxScaler(feature_range=(0,1))
C38U_scaler = MinMaxScaler(feature_range=(0,1))
E5H_scaler = MinMaxScaler(feature_range=(0,1))
U11_dataset = U11_scaler.fit_transform(U11_dataset[: , 1:]) #specify 1 to filter out date
U11_dataset = np.nan_to_num(U11_dataset)
C38U_dataset = C38U_scaler.fit_transform(C38U_dataset[: , 1:]) #specify 1 to filter out date
C38U_dataset = np.nan_to_num(C38U_dataset)
E5H_dataset = E5H_scaler.fit_transform(E5H_dataset[: , 1:]) #specify 1 to filter out date
E5H_dataset = np.nan_to_num(E5H_dataset)

#60% trainset, 40% testset
U11_trainset_split = int(0.6*len(U11_dataset))
U11_trainset = U11_dataset[0:U11_trainset_split, :]
U11_testset = U11_dataset[U11_trainset_split:len(U11_dataset), :]
C38U_trainset_split = int(0.6*len(C38U_dataset))
C38U_trainset = C38U_dataset[0:C38U_trainset_split, :]
C38U_testset = C38U_dataset[C38U_trainset_split:len(U11_dataset), :]
E5H_trainset_split = int(0.6*len(E5H_dataset))
E5H_trainset = E5H_dataset[0:E5H_trainset_split, :]
E5H_testset = E5H_dataset[E5H_trainset_split:len(U11_dataset), :]

#Prepare data
prediction_feature_colnum = 5

def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, prediction_feature_colnum])
    return np.array(dataX), np.array(dataY)


#how big is the window, in this case window size 1
# X=t, Y=t+1
look_back = 1
U11_trainX, U11_trainY = create_dataset(U11_trainset, look_back)
U11_testX, U11_testY = create_dataset(U11_testset, look_back)
C38U_trainX, C38U_trainY = create_dataset(C38U_trainset, look_back)
C38U_testX, C38U_testY = create_dataset(C38U_testset, look_back)
E5H_trainX, E5H_trainY = create_dataset(E5H_trainset, look_back)
E5H_testX, E5H_testY = create_dataset(E5H_testset, look_back)

U11_num_input_feature = 17
C38U_num_input_feature = 12
E5H_num_input_feature = 13

# reshape to -> [samples, timesteps, features]
U11_trainX = np.reshape(U11_trainX, (U11_trainX.shape[0], look_back, U11_num_input_feature))
U11_testX = np.reshape(U11_testX, (U11_testX.shape[0], look_back, U11_num_input_feature))
C38U_trainX = np.reshape(C38U_trainX, (C38U_trainX.shape[0], look_back, C38U_num_input_feature))
C38U_testX = np.reshape(C38U_testX, (C38U_testX.shape[0], look_back, C38U_num_input_feature))
E5H_trainX = np.reshape(E5H_trainX, (E5H_trainX.shape[0], look_back, E5H_num_input_feature))
E5H_testX = np.reshape(E5H_testX, (E5H_testX.shape[0], look_back, E5H_num_input_feature))

#build model
U11_model = Sequential()
U11_model.add(LSTM(128, input_shape=(look_back,U11_num_input_feature), return_sequences = True))
U11_model.add(LSTM(64, input_shape=(look_back,U11_num_input_feature), return_sequences = False))
U11_model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
U11_model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
U11_model.compile(loss = 'mse', optimizer = 'rmsprop')
U11_checkpoint = ModelCheckpoint('./U11_best_weights_model.hdf5', monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
U11_model_train = U11_model.fit(U11_trainX, U11_trainY, batch_size = len(U11_trainX),
                                  epochs = 10000, validation_split=0.30, shuffle = False,
                                  callbacks = [U11_checkpoint])
U11_model.load_weights('./U11_best_weights_model.hdf5')
U11_model.compile(loss = 'mse', optimizer = 'rmsprop')

C38U_model = Sequential()
C38U_model.add(LSTM(128, input_shape=(look_back,C38U_num_input_feature), return_sequences = True))
C38U_model.add(LSTM(64, input_shape=(look_back,C38U_num_input_feature), return_sequences = False))
C38U_model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
C38U_model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
C38U_model.compile(loss = 'mse', optimizer = 'rmsprop')
C38U_checkpoint = ModelCheckpoint('./C38U_best_weights_model.hdf5', monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
C38U_model_train = C38U_model.fit(C38U_trainX, C38U_trainY, batch_size = len(C38U_trainX),
                                  epochs = 10000, validation_split=0.30, shuffle = False,
                                  callbacks = [C38U_checkpoint])
C38U_model.load_weights('./C38U_best_weights_model.hdf5')
C38U_model.compile(loss = 'mse', optimizer = 'rmsprop')

E5H_model = Sequential()
E5H_model.add(LSTM(128, input_shape=(look_back,E5H_num_input_feature), return_sequences = True))
E5H_model.add(LSTM(64, input_shape=(look_back,E5H_num_input_feature), return_sequences = False))
E5H_model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
E5H_model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
E5H_model.compile(loss = 'mse', optimizer = 'rmsprop')
E5H_checkpoint = ModelCheckpoint('./E5H_best_weights_model.hdf5', monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
E5H_model_train = E5H_model.fit(E5H_trainX, E5H_trainY, batch_size = len(E5H_trainX),
                                epochs = 10000, validation_split=0.30, shuffle = False,
                                callbacks = [E5H_checkpoint])
E5H_model.load_weights('./E5H_best_weights_model.hdf5')
E5H_model.compile(loss = 'mse', optimizer = 'rmsprop')

#make predictions
U11_trainPredict = U11_model.predict(U11_trainX)
U11_testPredict = U11_model.predict(U11_testX)
C38U_trainPredict = C38U_model.predict(C38U_trainX)
C38U_testPredict = C38U_model.predict(C38U_testX)
E5H_trainPredict = E5H_model.predict(E5H_trainX)
E5H_testPredict = E5H_model.predict(E5H_testX)

#transform back to original scale
U11_trainPredict_extended = np.zeros((len(U11_trainPredict), U11_num_input_feature))
U11_trainPredict_extended[:, prediction_feature_colnum] = U11_trainPredict[:, 0]
U11_trainPredict = U11_scaler.inverse_transform(U11_trainPredict_extended)[:, prediction_feature_colnum]
U11_testPredict_extended = np.zeros((len(U11_testPredict), U11_num_input_feature))
U11_testPredict_extended[:, prediction_feature_colnum] = U11_testPredict[:, 0]
U11_testPredict = U11_scaler.inverse_transform(U11_testPredict_extended)[:, prediction_feature_colnum]

C38U_trainPredict_extended = np.zeros((len(C38U_trainPredict), C38U_num_input_feature))
C38U_trainPredict_extended[:, prediction_feature_colnum] = C38U_trainPredict[:, 0]
C38U_trainPredict = C38U_scaler.inverse_transform(C38U_trainPredict_extended)[:, prediction_feature_colnum]
C38U_testPredict_extended = np.zeros((len(C38U_testPredict), C38U_num_input_feature))
C38U_testPredict_extended[:, prediction_feature_colnum] = C38U_testPredict[:, 0]
C38U_testPredict = C38U_scaler.inverse_transform(C38U_testPredict_extended)[:, prediction_feature_colnum]

E5H_trainPredict_extended = np.zeros((len(E5H_trainPredict), E5H_num_input_feature))
E5H_trainPredict_extended[:, prediction_feature_colnum] = E5H_trainPredict[:, 0]
E5H_trainPredict = E5H_scaler.inverse_transform(E5H_trainPredict_extended)[:, prediction_feature_colnum]
E5H_testPredict_extended = np.zeros((len(E5H_testPredict), E5H_num_input_feature))
E5H_testPredict_extended[:, prediction_feature_colnum] = E5H_testPredict[:, 0]
E5H_testPredict = E5H_scaler.inverse_transform(E5H_testPredict_extended)[:, prediction_feature_colnum]

U11_trainY_extended = np.zeros((len(U11_trainY), U11_num_input_feature))
U11_trainY_extended[:, prediction_feature_colnum] = U11_trainY
U11_trainY = U11_scaler.inverse_transform(U11_trainY_extended)[:, prediction_feature_colnum]
U11_testY_extended = np.zeros((len(U11_testY), U11_num_input_feature))
U11_testY_extended[:, prediction_feature_colnum] = U11_testY
U11_testY = U11_scaler.inverse_transform(U11_testY_extended)[:, prediction_feature_colnum]

C38U_trainY_extended = np.zeros((len(C38U_trainY), C38U_num_input_feature))
C38U_trainY_extended[:, prediction_feature_colnum] = C38U_trainY
C38U_trainY = C38U_scaler.inverse_transform(C38U_trainY_extended)[:, prediction_feature_colnum]
C38U_testY_extended = np.zeros((len(C38U_testY), C38U_num_input_feature))
C38U_testY_extended[:, prediction_feature_colnum] = C38U_testY
C38U_testY = C38U_scaler.inverse_transform(C38U_testY_extended)[:, prediction_feature_colnum]

E5H_trainY_extended = np.zeros((len(E5H_trainY), E5H_num_input_feature))
E5H_trainY_extended[:, prediction_feature_colnum] = E5H_trainY
E5H_trainY = E5H_scaler.inverse_transform(E5H_trainY_extended)[:, prediction_feature_colnum]
E5H_testY_extended = np.zeros((len(E5H_testY), E5H_num_input_feature))
E5H_testY_extended[:, prediction_feature_colnum] = E5H_testY
E5H_testY = E5H_scaler.inverse_transform(E5H_testY_extended)[:, prediction_feature_colnum]

#caculate errors
U11_trainsetError = math.sqrt(mean_squared_error(U11_trainY, U11_trainPredict))
U11_trainsetErrorPercent = np.sqrt(np.mean(np.square(((U11_trainY - U11_trainPredict) /
                                                      U11_trainY)), axis=0))
U11_testsetError = math.sqrt(mean_squared_error(U11_testY, U11_testPredict))
U11_testsetErrorPercent = np.sqrt(np.mean(np.square(((U11_testY - U11_testPredict) /
                                                     U11_testY)), axis=0))
C38U_trainsetError = math.sqrt(mean_squared_error(C38U_trainY, C38U_trainPredict))
C38U_trainsetErrorPercent = np.sqrt(np.mean(np.square(((C38U_trainY - C38U_trainPredict) /
                                                      C38U_trainY)), axis=0))
C38U_testsetError = math.sqrt(mean_squared_error(C38U_testY, C38U_testPredict))
C38U_testsetErrorPercent = np.sqrt(np.mean(np.square(((C38U_testY - C38U_testPredict) /
                                                     C38U_testY)), axis=0))
E5H_trainsetError = math.sqrt(mean_squared_error(E5H_trainY, E5H_trainPredict))
E5H_trainsetErrorPercent = np.sqrt(np.mean(np.square(((E5H_trainY - E5H_trainPredict) /
                                                      E5H_trainY)), axis=0))
E5H_testsetError = math.sqrt(mean_squared_error(E5H_testY, E5H_testPredict))
E5H_testsetErrorPercent = np.sqrt(np.mean(np.square(((E5H_testY - E5H_testPredict) /
                                                     E5H_testY)), axis=0))
print()
print('U11.SI trainset error (RMSE): %.2f, Percentage error = %.4f' % (U11_trainsetError,
                                                                       U11_trainsetErrorPercent))
print('U11.SI testset error (RMSE): %.2f, Percentage error = %.4f' % (U11_testsetError,
                                                                      U11_testsetErrorPercent))
print('C38U.SI trainset error (RMSE): %.2f, Percentage error = %.4f' % (C38U_trainsetError,
                                                                        C38U_trainsetErrorPercent))
print('C38U.SI testset error (RMSE): %.2f, Percentage error = %.4f' % (C38U_testsetError,
                                                                       C38U_testsetErrorPercent))
print('E5H.SI trainset error (RMSE): %.2f, Percentage error = %.4f' % (E5H_trainsetError,
                                                                       E5H_trainsetErrorPercent))
print('E5H.SI testset error (RMSE): %.2f, Percentage error = %.4f' % (E5H_testsetError,
                                                                      E5H_testsetErrorPercent))

#prepare data to plot
U11_trainPredict_plot = np.empty_like(U11_dataset)
U11_trainPredict_plot[:, :] = np.nan
U11_trainPredict_plot[look_back:(len(U11_trainPredict) + look_back),
                      prediction_feature_colnum] = U11_trainPredict
U11_testPredict_plot = np.empty_like(U11_dataset)
U11_testPredict_plot[:, :] = np.nan
U11_testPredict_plot[(len(U11_trainPredict) + look_back + 1):
                     (len(U11_trainPredict) + look_back + 1 + len(U11_testPredict)),
                     prediction_feature_colnum] = U11_testPredict

C38U_trainPredict_plot = np.empty_like(C38U_dataset)
C38U_trainPredict_plot[:, :] = np.nan
C38U_trainPredict_plot[look_back:(len(C38U_trainPredict) + look_back),
                      prediction_feature_colnum] = C38U_trainPredict
C38U_testPredict_plot = np.empty_like(C38U_dataset)
C38U_testPredict_plot[:, :] = np.nan
C38U_testPredict_plot[(len(C38U_trainPredict) + look_back + 1):
                     (len(C38U_trainPredict) + look_back + 1 + len(C38U_testPredict)),
                     prediction_feature_colnum] = C38U_testPredict

E5H_trainPredict_plot = np.empty_like(E5H_dataset)
E5H_trainPredict_plot[:, :] = np.nan
E5H_trainPredict_plot[look_back:(len(E5H_trainPredict) + look_back),
                      prediction_feature_colnum] = E5H_trainPredict
E5H_testPredict_plot = np.empty_like(E5H_dataset)
E5H_testPredict_plot[:, :] = np.nan
E5H_testPredict_plot[(len(E5H_trainPredict) + look_back + 1):
                     (len(E5H_trainPredict) + look_back + 1 + len(E5H_testPredict)),
                     prediction_feature_colnum] = E5H_testPredict

#print out data
U11_dataset = U11_scaler.inverse_transform(U11_dataset)
C38U_dataset = C38U_scaler.inverse_transform(C38U_dataset)
E5H_dataset = E5H_scaler.inverse_transform(E5H_dataset)

U11_table = []
U11_table.append(['Date', 'Predicted', 'Trainset/Testset', 'Actual'])
C38U_table = []
C38U_table.append(['Date', 'Predicted', 'Trainset/Testset', 'Actual'])
E5H_table = []
E5H_table.append(['Date', 'Predicted', 'Trainset/Testset', 'Actual'])

for i in range(len(U11_dataset)):
    content = []
    if np.isnan(U11_trainPredict_plot[i][prediction_feature_colnum]):
        if np.isnan(U11_testPredict_plot[i][prediction_feature_colnum]):
            content.append(U11_date[i])
            content.append('-')
            content.append('-')
            content.append(U11_dataset[i][prediction_feature_colnum])
        else:
            content.append(U11_date[i])
            content.append(U11_testPredict_plot[i][prediction_feature_colnum])
            content.append('Testset')
            content.append(U11_dataset[i][prediction_feature_colnum])
    else:
        content.append(U11_date[i])
        content.append(U11_trainPredict_plot[i][prediction_feature_colnum])
        content.append('Trainset')
        content.append(U11_dataset[i][prediction_feature_colnum])
    U11_table.append(content)
for i in range(len(C38U_dataset)):
    content = []
    if np.isnan(C38U_trainPredict_plot[i][prediction_feature_colnum]):
        if np.isnan(C38U_testPredict_plot[i][prediction_feature_colnum]):
            content.append(C38U_date[i])
            content.append('-')
            content.append('-')
            content.append(C38U_dataset[i][prediction_feature_colnum])
        else:
            content.append(C38U_date[i])
            content.append(C38U_testPredict_plot[i][prediction_feature_colnum])
            content.append('Testset')
            content.append(C38U_dataset[i][prediction_feature_colnum])
    else:
        content.append(C38U_date[i])
        content.append(C38U_trainPredict_plot[i][prediction_feature_colnum])
        content.append('Trainset')
        content.append(C38U_dataset[i][prediction_feature_colnum])
    C38U_table.append(content)
for i in range(len(E5H_dataset)):
    content = []
    if np.isnan(E5H_trainPredict_plot[i][prediction_feature_colnum]):
        if np.isnan(E5H_testPredict_plot[i][prediction_feature_colnum]):
            content.append(E5H_date[i])
            content.append('-')
            content.append('-')
            content.append(E5H_dataset[i][prediction_feature_colnum])
        else:
            content.append(E5H_date[i])
            content.append(E5H_testPredict_plot[i][prediction_feature_colnum])
            content.append('Testset')
            content.append(E5H_dataset[i][prediction_feature_colnum])
    else:
        content.append(E5H_date[i])
        content.append(E5H_trainPredict_plot[i][prediction_feature_colnum])
        content.append('Trainset')
        content.append(E5H_dataset[i][prediction_feature_colnum])
    E5H_table.append(content)

print()

## printing results in python and exporting to excel
import csv

print('United Overseas Bank Limited (U11.SI) stock price prediction')
with open("UOB - Actual and Prediction.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(U11_table)):
        writer.writerow(U11_table[i])
        print(U11_table[i])

print()
print('CapitaLand Mall Trust (C38U.SI) stock price prediction')
with open("CMT - Actual and Prediction.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(C38U_table)):
        writer.writerow(C38U_table[i])
        print(C38U_table[i])

print()
print('Golden Agri-Resources Ltd (E5H.SI) (E5H.SI) stock price prediction')
with open("GAR - Actual and Prediction.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(E5H_table)):
        writer.writerow(E5H_table[i])
        print(E5H_table[i])

print()

#plot graph now
U11_plot = plt.figure(1)
U11_actual_data, = plt.plot(U11_dataset[:, prediction_feature_colnum], linestyle = '-')
#U11_trainset_prediction, = plt.plot(U11_trainPredict_plot[:, prediction_feature_colnum]
#                                  , linestyle = '--')
U11_testset_prediction, = plt.plot(U11_testPredict_plot[:, prediction_feature_colnum]
                                 , linestyle = '-.')
plt.title('United Overseas Bank Limited (U11.SI) stock price prediction')
plt.ylabel('Stock price (S$)')
plt.xlabel('Timestamp')
plt.legend([U11_actual_data, U11_testset_prediction],
           ['actual', 'testset prediction'], loc = 'upper left')

C38U_plot = plt.figure(2)
C38U_actual_data, = plt.plot(C38U_dataset[:, prediction_feature_colnum], linestyle = '-')
#C38U_trainset_prediction, = plt.plot(C38U_trainPredict_plot[:, prediction_feature_colnum]
#                                  , linestyle = '--')
C38U_testset_prediction, = plt.plot(C38U_testPredict_plot[:, prediction_feature_colnum]
                                 , linestyle = '-.')
plt.title('CapitaLand Mall Trust (C38U.SI) stock price prediction')
plt.ylabel('Stock price (S$)')
plt.xlabel('Timestamp')
plt.legend([C38U_actual_data, C38U_testset_prediction],
           ['actual', 'testset prediction'], loc = 'upper left')

E5H_plot = plt.figure(3)
E5H_actual_data, = plt.plot(E5H_dataset[:, prediction_feature_colnum], linestyle = '-')
#E5H_trainset_prediction, = plt.plot(E5H_trainPredict_plot[:, prediction_feature_colnum]
#                                  , linestyle = '--')
E5H_testset_prediction, = plt.plot(E5H_testPredict_plot[:, prediction_feature_colnum]
                                 , linestyle = '-.')
plt.title('Golden Agri-Resources Ltd (E5H.SI) (E5H.SI) stock price prediction')
plt.ylabel('Stock price (S$)')
plt.xlabel('Timestamp')
plt.legend([E5H_actual_data, E5H_testset_prediction],
           ['actual', 'testset prediction'], loc = 'upper left')

plt.show()

