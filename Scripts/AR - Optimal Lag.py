import pandas as pd
import matplotlib.pylab as plt
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import r2_score

## UOB (AR auto selection of lag value)
UOB_data = pd.read_csv('U11.SI weekly.csv', usecols = ['Date', 'Average'])

UOB_data['Date'] = pd.to_datetime(UOB_data['Date'])
UOB_data.set_index('Date', inplace=True)
print(UOB_data.head())

# stationary model (removed trend and seasonality) to account for auto correlation
UOB_data['stationary'] = UOB_data['Average'].diff()
#print(UOB_data)

# split dataset, data from 2019 onwards = testset
UOB_series = UOB_data['stationary'].dropna()
UOB_train_series = UOB_series[1:len(UOB_series)-9]
UOB_test_series = UOB_series[UOB_series[len(UOB_series)-9:]]
#print(UOB_train_series)
#print(UOB_test_series)

# train autoregression
model = AR(UOB_train_series)
model_fit = model.fit()
print('Lag value to eliminate autocorrelation for UOB: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

## CapitaLand Mall Trust
## CMT (AR auto selection of lag value)
CMT_data = pd.read_csv('C38U.SI weekly.csv', usecols = ['Date', 'Average'])

CMT_data['Date'] = pd.to_datetime(CMT_data['Date'])
CMT_data.set_index('Date', inplace=True)
print(CMT_data.head())

# stationary model (removed trend and seasonality) to account for auto correlation
CMT_data['stationary'] = CMT_data['Average'].diff()
#print(CMT_data)

# split dataset, data from 2019 onwards = testset
CMT_series = CMT_data['stationary'].dropna()
CMT_train_series = CMT_series[1:len(CMT_series)-9]
CMT_test_series = CMT_series[CMT_series[len(CMT_series)-9:]]
#print(CMT_train_series)
#print(CMT_test_series)

# train autoregression
model = AR(CMT_train_series)
model_fit = model.fit()
print('Lag value to eliminate autocorrelation for Capitaland: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

##Golden Agri-Resource
## GAR (AR auto selection of lag value)
GAR_data = pd.read_csv('E5H.SI weekly.csv', usecols = ['Date', 'Average'])

GAR_data['Date'] = pd.to_datetime(GAR_data['Date'])
GAR_data.set_index('Date', inplace=True)
print(GAR_data.head())

# stationary model (removed trend and seasonality) to account for auto correlation
GAR_data['stationary'] = GAR_data['Average'].diff()
#print(GAR_data)

# split dataset, data from 2019 onwards = testset
GAR_series = GAR_data['stationary'].dropna()
GAR_train_series = GAR_series[1:len(GAR_series)-9]
GAR_test_series = GAR_series[GAR_series[len(GAR_series)-9:]]
#print(GAR_train_series)
#print(GAR_test_series)

# train autoregression
model = AR(GAR_train_series)
model_fit = model.fit()
print('Lag value to eliminate autocorrelation for Golden Agri: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

print('Lag value to eliminate autocorrelation for UOB: %s' % model_fit.k_ar)
print('Lag value to eliminate autocorrelation for Capitaland: %s' % model_fit.k_ar)
print('Lag value to eliminate autocorrelation for Golden Agri: %s' % model_fit.k_ar)