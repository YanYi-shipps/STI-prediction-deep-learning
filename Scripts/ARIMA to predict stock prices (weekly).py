#LSTM loss function
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(2019)

U11_dataset = pd.read_csv('./U11.SI weekly.csv', header = 0)
C38U_dataset = pd.read_csv('./C38U.SI weekly.csv', header = 0)
E5H_dataset = pd.read_csv('./E5H.SI weekly.csv', header = 0)

prediction_feature_colnum = 5

U11_dataset = U11_dataset.values
C38U_dataset = C38U_dataset.values
E5H_dataset = E5H_dataset.values

U11_date = U11_dataset[: , 0] #first column is date, save it first 
C38U_date = C38U_dataset[: , 0] #first column is date, save it first
E5H_date = E5H_dataset[: , 0] #first column is date, save it first

U11_scaler = MinMaxScaler(feature_range=(0,1))
C38U_scaler = MinMaxScaler(feature_range=(0,1))
E5H_scaler = MinMaxScaler(feature_range=(0,1))
U11_dataset = U11_scaler.fit_transform(U11_dataset[: , prediction_feature_colnum+1].reshape(-1,1)) #specify 1 to filter out date
C38U_dataset = C38U_scaler.fit_transform(C38U_dataset[: , prediction_feature_colnum+1].reshape(-1,1)) #specify 1 to filter out date
E5H_dataset = E5H_scaler.fit_transform(E5H_dataset[: , prediction_feature_colnum+1].reshape(-1,1)) #specify 1 to filter out date

#row number 262 onwards in testset
#also change to only one variable
U11_trainset_split = int(0.6*len(U11_dataset))
U11_trainset = U11_dataset[0:U11_trainset_split]
U11_testset = U11_dataset[U11_trainset_split:len(U11_dataset)]
C38U_trainset_split = int(0.6*len(C38U_dataset))
C38U_trainset = C38U_dataset[0:C38U_trainset_split]
C38U_testset = C38U_dataset[C38U_trainset_split:len(U11_dataset)]
E5H_trainset_split = int(0.6*len(E5H_dataset))
E5H_trainset = E5H_dataset[0:E5H_trainset_split]
E5H_testset = E5H_dataset[E5H_trainset_split:len(U11_dataset)]

U11_seendata = [U11_trainset[x] for x in range(len(U11_trainset))]
C38U_seendata = [C38U_trainset[x] for x in range(len(C38U_trainset))]
E5H_seendata = [E5H_trainset[x] for x in range(len(E5H_trainset))]

U11_testPredict = []
C38U_testPredict = []
E5H_testPredict = []
#build model and predict

for timepoint in range(len(U11_testset)):
    actual = U11_testset[timepoint]
    model = ARIMA(U11_seendata, order=(15,1,0))
    model_fit = model.fit(disp=0)
    predicted = model_fit.forecast()[0]
    U11_testPredict.append(predicted)
    U11_seendata.append(actual)

for timepoint in range(len(C38U_testset)):
    actual = C38U_testset[timepoint]
    model = ARIMA(C38U_seendata, order=(2,1,0))
    model_fit = model.fit(disp=0)
    predicted = model_fit.forecast()[0]
    C38U_testPredict.append(predicted)
    C38U_seendata.append(actual)

for timepoint in range(len(E5H_testset)):
    actual = E5H_testset[timepoint]
    model = ARIMA(E5H_seendata, order=(2,1,0))
    model_fit = model.fit(disp=0)
    predicted = model_fit.forecast()[0]
    E5H_testPredict.append(predicted)
    E5H_seendata.append(actual)

#reshape data
U11_testPredict = np.array(U11_testPredict).reshape(-1,1)
C38U_testPredict = np.array(C38U_testPredict).reshape(-1,1)
E5H_testPredict = np.array(E5H_testPredict).reshape(-1,1)

#transform back to original scale
U11_testPredict_extended = np.zeros(len(U11_testPredict))
U11_testPredict_extended = U11_testPredict
U11_testPredict = U11_scaler.inverse_transform(U11_testPredict_extended)

C38U_testPredict_extended = np.zeros(len(C38U_testPredict))
C38U_testPredict_extended = C38U_testPredict
C38U_testPredict = C38U_scaler.inverse_transform(C38U_testPredict_extended)

E5H_testPredict_extended = np.zeros(len(E5H_testPredict))
E5H_testPredict_extended = E5H_testPredict
E5H_testPredict = E5H_scaler.inverse_transform(E5H_testPredict_extended)

U11_testset = U11_scaler.inverse_transform(U11_testset)
C38U_testset = C38U_scaler.inverse_transform(C38U_testset)
E5H_testset = E5H_scaler.inverse_transform(E5H_testset)

U11_dataset = U11_scaler.inverse_transform(U11_dataset)
C38U_dataset = C38U_scaler.inverse_transform(C38U_dataset)
E5H_dataset = E5H_scaler.inverse_transform(E5H_dataset)

#caculate errors
U11_testsetError = math.sqrt(mean_squared_error(U11_testset, U11_testPredict))
U11_testsetErrorPercent = np.sqrt(np.mean(np.square(((U11_testset - U11_testPredict) /
                                                      U11_testset)), axis=0))
C38U_testsetError = math.sqrt(mean_squared_error(C38U_testset, C38U_testPredict))
C38U_testsetErrorPercent = np.sqrt(np.mean(np.square(((C38U_testset - C38U_testPredict) /
                                                     C38U_testset)), axis=0))
E5H_testsetError = math.sqrt(mean_squared_error(E5H_testset, E5H_testPredict))
E5H_testsetErrorPercent = np.sqrt(np.mean(np.square(((E5H_testset - E5H_testPredict) /
                                                     E5H_testset)), axis=0))
print()
print('U11.SI testset error (RMSE): %.2f, Percentage error = %.4f' % (U11_testsetError,
                                                                      U11_testsetErrorPercent))

print('C38U.SI testset error (RMSE): %.2f, Percentage error = %.4f' % (C38U_testsetError,
                                                                       C38U_testsetErrorPercent))
print('E5H.SI testset error (RMSE): %.2f, Percentage error = %.4f' % (E5H_testsetError,
                                                                      E5H_testsetErrorPercent))

#prepare data to plot
U11_testPredict_plot = np.empty_like(U11_dataset)
U11_testPredict_plot[:] = np.nan
U11_testPredict_plot[U11_trainset_split:len(U11_dataset)] = U11_testPredict

C38U_testPredict_plot = np.empty_like(C38U_dataset)
C38U_testPredict_plot[:] = np.nan
C38U_testPredict_plot[C38U_trainset_split:len(C38U_dataset)] = C38U_testPredict

E5H_testPredict_plot = np.empty_like(E5H_dataset)
E5H_testPredict_plot[:] = np.nan
E5H_testPredict_plot[E5H_trainset_split:len(E5H_dataset)] = E5H_testPredict

#plot graph now
U11_plot = plt.figure(1)
U11_actual_data, = plt.plot(U11_dataset[:], linestyle = '-')
U11_testset_prediction, = plt.plot(U11_testPredict_plot[:]
                                 , linestyle = '-.')
plt.title('United Overseas Bank Limited (U11.SI) stock price prediction')
plt.ylabel('Stock price (S$)')
plt.xlabel('Timestamp')
plt.legend([U11_actual_data, U11_testset_prediction],
           ['actual', 'testset prediction'], loc = 'upper left')

C38U_plot = plt.figure(2)
C38U_actual_data, = plt.plot(C38U_dataset[:], linestyle = '-')
C38U_testset_prediction, = plt.plot(C38U_testPredict_plot[:]
                                 , linestyle = '-.')
plt.title('CapitaLand Mall Trust (C38U.SI) stock price prediction')
plt.ylabel('Stock price (S$)')
plt.xlabel('Timestamp')
plt.legend([C38U_actual_data, C38U_testset_prediction],
           ['actual', 'testset prediction'], loc = 'upper left')

E5H_plot = plt.figure(3)
E5H_actual_data, = plt.plot(E5H_dataset[:], linestyle = '-')
E5H_testset_prediction, = plt.plot(E5H_testPredict_plot[:]
                                 , linestyle = '-.')
plt.title('Golden Agri-Resources Ltd (E5H.SI) (E5H.SI) stock price prediction')
plt.ylabel('Stock price (S$)')
plt.xlabel('Timestamp')
plt.legend([E5H_actual_data, E5H_testset_prediction],
           ['actual', 'testset prediction'], loc = 'upper left')

plt.show()
