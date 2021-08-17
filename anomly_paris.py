# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Importing Lib for Ploting
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from datetime import datetime 

# Import Data Set
total_df = pd.read_csv("paris_dataset.csv", index_col=0)
total_df.set_index('DateTime', inplace=True)       

"""_____________________________________Feature Engineering____________________________"""




#Feature Scaling of independent Variables
from sklearn.preprocessing import MinMaxScaler
iv_columns = ['outdoor_humidity', 'outdoor_temperature', 'wind_speed',
                      'App-1', 'App-2', 'App-3']
iv_transformer = MinMaxScaler(feature_range= (0,1))
iv_transformer = iv_transformer.fit(total_df[iv_columns].to_numpy())
total_df[iv_columns] = iv_transformer.transform(total_df[iv_columns].to_numpy())

#Feature Scaling of Dependent Variables
dv_columns = ['Units']
dv_transformer = MinMaxScaler(feature_range= (0,1))
dv_transformer = dv_transformer.fit(total_df[dv_columns])
total_df[dv_columns] = dv_transformer.transform(total_df[dv_columns])              




"""_____________________________________Test/Train Dataset split ____________________________"""

#Function to Split dataset into train and test set
def split_dataset(dataset, split_factor):
    train_size = int(len(dataset) * split_factor)
    test_size = len(dataset) - train_size
    train, test = dataset.iloc[0:train_size], dataset.iloc[train_size:len(dataset)]
    print(len(train), len(test))
    return train , test

#Split Dataset
dataset = total_df
split_factor = 0.95
train , test = split_dataset(dataset ,split_factor)






"""_____________________________________Look back Function for Single Output______________________"""

#Look Back function
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24*5
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train.Units, time_steps )
X_test , y_test  = create_dataset(test , test.Units , time_steps )
print(X_train.shape, y_train.shape)



"""___________________________________LSTM Framework____________________________________"""

#LSTM AutoEncoder
#Typer of RNN that tries to reconstruct itself
#If error is above threshold or below it that exhibits an anomaly

#Importing the keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM , TimeDistributed
from tensorflow.keras.layers import Dropout , RepeatVector

#Initializing the RNN
regressor = Sequential()

#Adding the LSTM layer and some Dropout Regularization
regressor.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
regressor.add(LSTM(32, activation='relu', return_sequences=False))
#Adding the Repeat Vector
regressor.add(RepeatVector(n = X_train.shape[1]))

regressor.add(LSTM(32, activation='relu', return_sequences=True))
regressor.add(LSTM(64, activation='relu', return_sequences=True))
regressor.add(TimeDistributed(Dense(X_train.shape[2])))





"""___________________________________Fitting and Prediction____________________________________"""
#Compiling the RNN
regressor.compile(optimizer= 'adam',loss= 'mae')
regressor.summary()



#Fitting the RNN to the training set
history = regressor.fit(X_train , y_train ,batch_size = 120 ,  
                        epochs = 3, validation_split=0.1, shuffle=False)


X_train_pred = regressor.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred,X_train),axis = 1)
train_mae_loss.shape

sns.displot(train_mae_loss,bins = 50, kde = True)
#Setting threshold and defining anamoly
X_test_pred = regressor.predict(X_test)



test_mae_loss = np.mean(np.abs(X_test_pred,X_test), axis = 1)

THRESHOLD = 1.2

test_score_df = pd.DataFrame(index=test[time_steps:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['Global_active_power'] = test[time_steps:].Global_active_power

plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend()

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()

plt.plot(
  test[time_steps:].index, 
  cnt_transformer.inverse_transform(test[time_steps:].close), 
  label='close price'
);

sns.scatterplot(
  anomalies.index,
  cnt_transformer.inverse_transform(anomalies.close),
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();
