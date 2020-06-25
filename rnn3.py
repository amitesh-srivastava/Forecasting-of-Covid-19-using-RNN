import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.dates as mdates

dataset = pd.read_excel('WHO1.xlsx')
dataset = dataset.loc[dataset['Country_code']=='LK']
#dataset = dataset.drop('Cumulative Deaths',axis=1)
data = dataset.iloc[:,5:6]
training_set = data[0:121]
#test_set = data[75:83]

ch=118

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(ch,121):
    X_train.append(training_set_scaled[i-ch:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


regressor.fit(X_train, y_train, epochs = 200, batch_size = 5)


pred = []
op5=[]

for k in range(40):
    inputs = training_set[len(training_set) - ch:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(ch,ch+1):
        X_test.append(inputs[i-ch:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted = regressor.predict(X_test)
    predicted = sc.inverse_transform(predicted)
    pred.append(predicted)
    df2 = pd.DataFrame({'Cumulative_cases': predicted[:, 0]})
    training_set = training_set.append(df2, ignore_index=True)
    op5.append(pred[k][0])


trs = training_set.iloc[:,0:1].values

op2=[]

dates=[]
for i in range(27,32):
    da = str(i)+"/05/2020"
    dates.append(da)

for i in range(1,31):
    da = str(i)+"/06/2020"
    dates.append(da)
for i in range(1,6):
    da = str(i)+"/07/2020"
    dates.append(da)

x_values = [datetime.datetime.strptime(d,"%d/%m/%Y").date() for d in dates]


ax = plt.gca()
formatter = mdates.DateFormatter("%Y-%m-%d")

ax.xaxis.set_major_formatter(formatter)

locator = mdates.DayLocator()

ax.xaxis.set_major_locator(locator)    

#plt.plot(trs, color = 'red', label = 'Predicted Deaths')
plt.plot(x_values,op5, color = 'blue', label = 'Prediction of Confirmed Cases')
plt.title('COVID 19 Confirmed Cases prediction')
plt.xlabel('Dates')
plt.ylabel('Number of People')
plt.legend()
plt.tick_params(axis='x', which='major', labelsize=9)
plt.tight_layout()