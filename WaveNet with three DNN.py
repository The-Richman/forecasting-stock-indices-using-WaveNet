# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:12:18 2021

@author: Hengde Wang @ University of Southampton
Student ID: 31541011 Email: hw2n20@soton.ac.uk
"""
#%% Import
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis, skew
#%%Load data 
df_csi300 = pd.read_csv('D:/MyCode/Dissertation/SCI300 index.csv',header = 0,index_col=(0),parse_dates=True,squeeze=True)
df_exchange = pd.read_excel('D:/MyCode/Dissertation/exchange rates.xlsx',header = 0,index_col=(0),parse_dates=True,squeeze=True,sheet_name = 'Sheet0',usecols = [0,1,2,3,4]).loc['2020-12-31':'2005-04-08',:]
df_exchange = df_exchange.sort_index(ascending = True)
df_volume = pd.read_csv('D:/MyCode/Dissertation/csi300 volume.csv',header = 0,index_col=(0),parse_dates=True,squeeze=True, usecols=[0,5])
df_volume = df_volume.sort_index(ascending = True)
df_volume = df_volume.str.replace('K','').astype(float)
df_exchange = df_exchange.astype(float)
df_exchange.columns = ['USD','EUR','JPY','HKD']
diff_index = df_exchange.index.difference(df_csi300.index)
diff_index2 = df_csi300.index.difference(df_exchange.index)
#%%Data vasualization
plt.subplot(221)
df_csi300.Closing.plot(color='coral', legend= True)
plt.xticks([])
plt.subplot(222)
plt.xticks([])
plt.yticks([])
df_csi300.Open.plot(color='skyblue',legend= True)
plt.subplot(223)
df_csi300.Hign.plot(color='burlywood',legend= True)
plt.subplot(224)
df_csi300.Low.plot(color='yellowgreen',legend= True)
plt.yticks([])
plt.subplots_adjust(wspace=0,hspace=0)
plt.suptitle('CSI300 Index')
plt.show()
plt.subplot(221)
df_exchange.USD.plot(color='coral', legend= True)
plt.xticks([])
plt.subplot(222)
df_exchange.EUR.plot(color='skyblue',legend= True)
plt.xticks([])
plt.subplot(223)
df_exchange.JPY.plot(color='burlywood',legend= True)
plt.xlabel('Dates')
plt.subplot(224)
df_exchange.HKD.plot(color='yellowgreen',legend= True)
plt.xlabel('Dates')
plt.subplots_adjust(hspace=0)
plt.suptitle('Exchange Rates')
plt.show()
df_volume.plot()
plt.ylabel('Trading volumes(K)')
plt.xlabel('Dates')
plt.show()
#%%Show statistics
print(df_csi300.describe())
print(df_exchange.describe())
print(df_volume.describe())

for item in diff_index:
    print(item)

#%%Deleted different dates
for item in diff_index:
    try:
        df_csi300.drop(item,inplace = True)
    except:
        print('No Corresponding dates')
for item in diff_index:
    try:
        df_exchange.drop(item,inplace = True)  
    except:
        print('No Corresponding dates')
for item in diff_index2:
    try:
        df_exchange.drop(item,inplace = True)  
    except:
        print('No Corresponding dates')
for item in diff_index2:
    try:
        df_csi300.drop(item,inplace = True)  
    except:
        print('No Corresponding dates')
#%% Clean
diff_index3 = df_volume.index.difference(df_exchange.index)
for item in diff_index3:
    try:
        df_volume.drop(item,inplace = True)  
    except:
        print('No Corresponding dates')
#%%
print("CSI 300 skew : ",skew(df_csi300.iloc[4:,]))
print("CSI 300 kurt : ",kurtosis(df_csi300.iloc[4:,]))
print("Exchange rates 300 skew : ",skew(df_exchange.iloc[4:,]))
print("Exchange rates 300 kurt : ",kurtosis(df_exchange.iloc[4:,]))
print("Volume 300 skew : ",skew(df_volume[4:]))
print("Volume 300 kurt : ",kurtosis(df_volume[4:]))
#%%
csi300 = df_csi300.copy().to_numpy()
exchange = df_exchange.copy().to_numpy()
volume = df_volume.copy().to_numpy()
volume = volume.reshape(volume.shape[0],1)
inputs = np.hstack((csi300,exchange))
inputs = np.hstack((inputs,volume))
org_train = inputs.copy()[:2325]
org_valid = inputs.copy()[2325:3074]
org_test = inputs.copy()[3074:]

#org_train = inputs.copy()[4:2344]
#org_valid = inputs.copy()[2344:3084]
#org_test = inputs.copy()[3084:]
#%%Preprocessing
#min-max scaling Normalization
scaler = MinMaxScaler()
scaler.fit(org_train)
inputs_prece = scaler.transform(inputs)
#target = inputs[:,3].copy().reshape(3824,1)
#inputs_prece = np.hstack((inputs_prece,target))
#%%Spliting data 
#Split into training, validation, test set
n_steps_in = 50
n_steps_out = 5
n_features = inputs.shape[1]
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, Y = list(), list()
	for i in range(sequence.shape[0]):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > sequence.shape[0]:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix,:], sequence[end_ix:out_end_ix,3]
		X.append(seq_x)
		Y.append(seq_y)
	return np.array(X), np.array(Y)

X_train, Y_train = split_sequence(inputs_prece[:2324], n_steps_in, n_steps_out)
X_valid, Y_valid = split_sequence(inputs_prece[2324:3074], n_steps_in, n_steps_out)
X_test, Y_test = split_sequence(inputs_prece[3074:], n_steps_in, n_steps_out)
#%%Simple aritificial neuron network
ann = keras.models.Sequential([
#keras.layers.InputLayer(input_shape=[n_steps_in,n_features]),
keras.layers.Flatten(input_shape=[n_steps_in, n_features]),
keras.layers.Dense(50),
keras.layers.Dense(50),
keras.layers.Dense(n_steps_out)
])
ann.compile(optimizer='adam', loss='mse')
ann_his = ann.fit(X_train, Y_train, epochs=100,validation_data=(X_valid, Y_valid))

ann_pred = ann.predict(X_test)
mse_ann = mean_squared_error(Y_test, ann_pred)
#%% linear regression
X_li = X_train.copy()
X_li = X_li.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
linear2 = LinearRegression()
linear2.fit(X_li,Y_train)
linear_pred2 = linear2.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
mse_linear = mean_squared_error(Y_test, linear_pred2)
print(mse_linear)
#%%Hyperparameters tuning for LSTM
param_lstm = {
"n_hidden": [0, 1, 2, 3 ], #hidden layer range(1-4)
"n_neurons": np.arange(1, 101),
"learning_rate": reciprocal(3e-4, 3e-2),
}
def build_lstm(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[n_steps_in,n_features]):
    model = keras.models.Sequential()
    if n_hidden == 0:
        model.add(keras.layers.LSTM(n_neurons,input_shape = input_shape))
    else:    
        model.add(keras.layers.LSTM(n_neurons, return_sequences=True,input_shape=input_shape))
        for layer in range(n_hidden - 1):
            model.add(keras.layers.LSTM(n_neurons, return_sequences=True, activation="relu"))
        model.add(keras.layers.LSTM(n_neurons))
    model.add(keras.layers.Dense(n_steps_out))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_lstm = keras.wrappers.scikit_learn.KerasRegressor(build_lstm)
rnd_search_lstm = RandomizedSearchCV(keras_lstm, param_lstm, n_iter=1,scoring = 'neg_mean_squared_error',cv=3)
rnd_search_lstm.fit(X_train, Y_train, epochs=50,validation_data=(X_valid, Y_valid),batch_size = 32, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
#%% Random search results of LSTM 
best_lstm = rnd_search_lstm.best_params_
#{'learning_rate': 0.00045534239006987106, 'n_hidden': 0, 'n_neurons': 81}
cvres = rnd_search_lstm.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)     
#0.100396905801279 {'learning_rate': 0.0015959550386786402, 'n_hidden': 2, 'n_neurons': 26}
#0.15268053083462224 {'learning_rate': 0.0003471590310569915, 'n_hidden': 3, 'n_neurons': 5}
#0.09196843911250446 {'learning_rate': 0.001482672854411444, 'n_hidden': 3, 'n_neurons': 16}
#0.070881680185905 {'learning_rate': 0.0007545060613941868, 'n_hidden': 0, 'n_neurons': 11}
#0.07872796219113395 {'learning_rate': 0.0011007336538857455, 'n_hidden': 1, 'n_neurons': 83}
#0.12183338777437862 {'learning_rate': 0.003877298244392933, 'n_hidden': 2, 'n_neurons': 4}
#0.054821438805938434 {'learning_rate': 0.02275109303632742, 'n_hidden': 0, 'n_neurons': 54}
#0.05245277895323541 {'learning_rate': 0.007616705258632423, 'n_hidden': 1, 'n_neurons': 86}
#0.07679903975871036 {'learning_rate': 0.006881320042565835, 'n_hidden': 2, 'n_neurons': 25}
#0.06864382766422154 {'learning_rate': 0.009718916435360521, 'n_hidden': 2, 'n_neurons': 23}
#0.17710693176856843 {'learning_rate': 0.024175882926449326, 'n_hidden': 3, 'n_neurons': 55}
#0.13115132185878114 {'learning_rate': 0.0008380620566055429, 'n_hidden': 3, 'n_neurons': 20}
#0.16117643794224717 {'learning_rate': 0.006192467988696549, 'n_hidden': 3, 'n_neurons': 2}
#0.07361831455077811 {'learning_rate': 0.018823590639682145, 'n_hidden': 2, 'n_neurons': 33}
#0.07370649402421567 {'learning_rate': 0.0005206716916031646, 'n_hidden': 0, 'n_neurons': 81}
#0.07964817562632288 {'learning_rate': 0.019597845443263404, 'n_hidden': 1, 'n_neurons': 78}
#0.05262439440376345 {'learning_rate': 0.0007182718765725603, 'n_hidden': 0, 'n_neurons': 10}
#0.0551419288212945 {'learning_rate': 0.004656756667770108, 'n_hidden': 1, 'n_neurons': 54}
#0.09744944896076466 {'learning_rate': 0.018997037720508363, 'n_hidden': 3, 'n_neurons': 37}
#0.04471571140407716 {'learning_rate': 0.00045534239006987106, 'n_hidden': 0, 'n_neurons': 81}
#%%Define the path of LSTM
lstm_path = os.path.join('LSTM', '001')
#%% Save LSTM
model_lstm = rnd_search_lstm.best_estimator_.model
tf.saved_model.save(model_lstm, lstm_path)
#%%Load LSTM
model_lstm = tf.saved_model.load(lstm_path)
#%%Hyperparameters tuning for GRU
param_gru = {
"n_hidden": [0, 1, 2, 3 ], #hidden layer range(1-4)
"n_neurons": np.arange(1, 101),
"learning_rate": reciprocal(3e-4, 3e-2),
}
def build_gru(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[n_steps_in,n_features]):
    model = keras.models.Sequential()
    if n_hidden == 0:
        model.add(keras.layers.GRU(n_neurons,input_shape = input_shape))
    else:    
        model.add(keras.layers.GRU(n_neurons, return_sequences=True,input_shape=input_shape))
        for layer in range(n_hidden - 1):
            model.add(keras.layers.GRU(n_neurons, return_sequences=True, activation="relu"))
        model.add(keras.layers.GRU(n_neurons))
    model.add(keras.layers.Dense(n_steps_out))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
keras_gru = keras.wrappers.scikit_learn.KerasRegressor(build_gru)
rnd_search_gru = RandomizedSearchCV(keras_gru, param_gru, n_iter=20,scoring = 'neg_mean_squared_error',cv=3)
rnd_search_gru.fit(X_train, Y_train, epochs=50,validation_data=(X_valid, Y_valid),batch_size = 64, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
#%% results of gru models
best_gru = rnd_search_gru.best_params_
#{'learning_rate': 0.0036759864968854924, 'n_hidden': 0, 'n_neurons': 76}
cvres_gru = rnd_search_gru.cv_results_
for mean_score, params in zip(cvres_gru["mean_test_score"],cvres_gru["params"]):
    print(np.sqrt(-mean_score), params)
#0.03875797433200118 {'learning_rate': 0.009074437304858103, 'n_hidden': 3, 'n_neurons': 95}
#0.07847401968309059 {'learning_rate': 0.0015369897917264395, 'n_hidden': 2, 'n_neurons': 4}
#0.0647881843226102 {'learning_rate': 0.011373427100837857, 'n_hidden': 3, 'n_neurons': 61}
#0.08394671420351625 {'learning_rate': 0.015189219679382472, 'n_hidden': 3, 'n_neurons': 38}
#0.051212815510265276 {'learning_rate': 0.015027872899267653, 'n_hidden': 2, 'n_neurons': 21}
#0.043516184259643806 {'learning_rate': 0.008044230457889085, 'n_hidden': 2, 'n_neurons': 56}
#0.043665975696778346 {'learning_rate': 0.0011650498588671766, 'n_hidden': 2, 'n_neurons': 97}
#0.03604455056261402 {'learning_rate': 0.007215804674204915, 'n_hidden': 1, 'n_neurons': 80}
#0.026835902316743106 {'learning_rate': 0.007434661181754212, 'n_hidden': 0, 'n_neurons': 63}
#0.02675878968421125 {'learning_rate': 0.0020270137999979423, 'n_hidden': 0, 'n_neurons': 95}
#0.03958677524021475 {'learning_rate': 0.009820227993923793, 'n_hidden': 1, 'n_neurons': 10}
#0.07007474729793273 {'learning_rate': 0.011460859494850005, 'n_hidden': 3, 'n_neurons': 81}
#0.03669023932099481 {'learning_rate': 0.0006643556658154019, 'n_hidden': 0, 'n_neurons': 19}
#0.03410953047087134 {'learning_rate': 0.001962171296834118, 'n_hidden': 2, 'n_neurons': 42}
#0.06691860842132197 {'learning_rate': 0.022368131456458617, 'n_hidden': 3, 'n_neurons': 98}
#0.03448763783535831 {'learning_rate': 0.0009490711972321215, 'n_hidden': 0, 'n_neurons': 36}
#0.06258391324507247 {'learning_rate': 0.0049736842414437165, 'n_hidden': 3, 'n_neurons': 59}
#0.04279083325251431 {'learning_rate': 0.0014966288851411896, 'n_hidden': 1, 'n_neurons': 10}
#0.03286717975816598 {'learning_rate': 0.018276712534806963, 'n_hidden': 1, 'n_neurons': 59}
#0.029494261598775325 {'learning_rate': 0.017098660411754403, 'n_hidden': 1, 'n_neurons': 100}
#%%Define the path of GRU
gru_path = os.path.join('GRU', '001')
#%% Save GRU
model_gru = rnd_search_gru.best_estimator_.model
tf.saved_model.save(model_gru, gru_path)
#%%Load GRU
model_gru = tf.saved_model.load(gru_path)
#%% Random search of CNN
param_cnn = {
"n_hidden": [1, 2, 3],
"learning_rate": reciprocal(3e-4, 3e-2),
"n_filters":[1,2,4,8,16,32,64,128],
"kernel_size":[1,2,3,4]
}
def build_cnn(n_hidden=1,n_filters = 64, n_neurons=50,kernel_size=2, learning_rate=1e-3,input_shape=[n_steps_in,n_features]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for i in range(n_hidden):
        model.add(keras.layers.Conv1D(filters=n_filters,kernel_size=kernel_size,activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(n_steps_out))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
keras_cnn = keras.wrappers.scikit_learn.KerasRegressor(build_cnn)
rnd_search_cnn = RandomizedSearchCV(keras_cnn, param_cnn, n_iter=40,scoring = 'neg_mean_squared_error',cv=3)
rnd_search_cnn.fit(X_train, Y_train, epochs=50,validation_data=(X_valid, Y_valid),batch_size = 16, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
#%% cnn results
best_cnn = rnd_search_cnn.best_params_
#{'kernel_size': 3,
#'learning_rate': 0.0030154687241528313,
#'n_filters': 4,
#'n_hidden': 2}
cvres_cnn = rnd_search_cnn.cv_results_
for mean_score, params in zip(cvres_cnn["mean_test_score"],cvres_cnn["params"]):
    print(np.sqrt(-mean_score), params)
'''
0.1897303079408834 {'kernel_size': 4, 'learning_rate': 0.0015024547516595092, 'n_filters': 2, 'n_hidden': 2}
0.06063679243237914 {'kernel_size': 3, 'learning_rate': 0.0031464719471366196, 'n_filters': 32, 'n_hidden': 2}
0.08202206806007696 {'kernel_size': 3, 'learning_rate': 0.0017264421048575934, 'n_filters': 2, 'n_hidden': 2}
0.18408509457780461 {'kernel_size': 3, 'learning_rate': 0.0043438502681272355, 'n_filters': 2, 'n_hidden': 1}
0.06632324559522786 {'kernel_size': 3, 'learning_rate': 0.011376017156721138, 'n_filters': 8, 'n_hidden': 1}
0.07779250380486483 {'kernel_size': 1, 'learning_rate': 0.0011510162816615126, 'n_filters': 16, 'n_hidden': 3}
0.10902923896004296 {'kernel_size': 2, 'learning_rate': 0.001515591220653229, 'n_filters': 128, 'n_hidden': 2}
0.08098966592442783 {'kernel_size': 3, 'learning_rate': 0.002274113985268721, 'n_filters': 32, 'n_hidden': 3}
0.1887265702541354 {'kernel_size': 4, 'learning_rate': 0.0024115412486040142, 'n_filters': 4, 'n_hidden': 3}
0.07381646952607217 {'kernel_size': 2, 'learning_rate': 0.011751906366385413, 'n_filters': 128, 'n_hidden': 2}
0.1614042198580018 {'kernel_size': 3, 'learning_rate': 0.0016650559130966136, 'n_filters': 2, 'n_hidden': 2}
0.08338161731869287 {'kernel_size': 4, 'learning_rate': 0.01461012613631209, 'n_filters': 8, 'n_hidden': 2}
0.1534885889100724 {'kernel_size': 2, 'learning_rate': 0.0004865361232197484, 'n_filters': 64, 'n_hidden': 3}
0.6143470172309721 {'kernel_size': 2, 'learning_rate': 0.011263817874038316, 'n_filters': 2, 'n_hidden': 2}
0.062362736906350885 {'kernel_size': 3, 'learning_rate': 0.005545053503855731, 'n_filters': 8, 'n_hidden': 1}
0.0704293725512972 {'kernel_size': 4, 'learning_rate': 0.007065354277061494, 'n_filters': 128, 'n_hidden': 2}
0.1942235273828428 {'kernel_size': 1, 'learning_rate': 0.0005248421400571189, 'n_filters': 2, 'n_hidden': 2}
0.06728583770342618 {'kernel_size': 3, 'learning_rate': 0.0025461903497047385, 'n_filters': 4, 'n_hidden': 2}
0.12259202543831854 {'kernel_size': 2, 'learning_rate': 0.0022868413453304578, 'n_filters': 128, 'n_hidden': 1}
0.09636330571393932 {'kernel_size': 1, 'learning_rate': 0.004337374566211301, 'n_filters': 64, 'n_hidden': 2}
0.10135073838299764 {'kernel_size': 2, 'learning_rate': 0.0014634599847264211, 'n_filters': 1, 'n_hidden': 2}
0.16967884946148737 {'kernel_size': 4, 'learning_rate': 0.00036583895921848595, 'n_filters': 2, 'n_hidden': 3}
0.09760921912636128 {'kernel_size': 4, 'learning_rate': 0.0007028542075948574, 'n_filters': 64, 'n_hidden': 1}
0.171362683985709 {'kernel_size': 2, 'learning_rate': 0.0011008105749570848, 'n_filters': 4, 'n_hidden': 1}
0.11161680374229033 {'kernel_size': 4, 'learning_rate': 0.0006460644180953533, 'n_filters': 32, 'n_hidden': 2}
0.20195985756943743 {'kernel_size': 2, 'learning_rate': 0.000982393106081153, 'n_filters': 1, 'n_hidden': 3}
0.037585750869823256 {'kernel_size': 3, 'learning_rate': 0.0030154687241528313, 'n_filters': 4, 'n_hidden': 2}
0.08077756202048757 {'kernel_size': 3, 'learning_rate': 0.003366731312869117, 'n_filters': 16, 'n_hidden': 1}
0.19043885849675374 {'kernel_size': 4, 'learning_rate': 0.009544031655179338, 'n_filters': 1, 'n_hidden': 1}
0.08206713751011299 {'kernel_size': 2, 'learning_rate': 0.000318370831697335, 'n_filters': 8, 'n_hidden': 1}
0.1004195192910636 {'kernel_size': 4, 'learning_rate': 0.0005806748370347655, 'n_filters': 16, 'n_hidden': 3}
0.11403704674133855 {'kernel_size': 2, 'learning_rate': 0.0017848584997021853, 'n_filters': 128, 'n_hidden': 3}
0.08678558348009445 {'kernel_size': 4, 'learning_rate': 0.0004001316058780453, 'n_filters': 2, 'n_hidden': 2}
0.07167017828471578 {'kernel_size': 4, 'learning_rate': 0.0010553220040113616, 'n_filters': 4, 'n_hidden': 3}
0.18920014337186197 {'kernel_size': 4, 'learning_rate': 0.001251822934166637, 'n_filters': 4, 'n_hidden': 2}
0.09528096538534563 {'kernel_size': 3, 'learning_rate': 0.005896860857247167, 'n_filters': 2, 'n_hidden': 1}
0.13667392478870294 {'kernel_size': 4, 'learning_rate': 0.0009236252175994836, 'n_filters': 16, 'n_hidden': 3}
0.2280453627797788 {'kernel_size': 2, 'learning_rate': 0.009284911982764035, 'n_filters': 2, 'n_hidden': 1}
0.16224267126141392 {'kernel_size': 2, 'learning_rate': 0.00112521162193705, 'n_filters': 8, 'n_hidden': 3}
0.06802168657368408 {'kernel_size': 1, 'learning_rate': 0.002528048304726661, 'n_filters': 4, 'n_hidden': 3}
'''
#%%Build CNN
CNN = build_cnn(kernel_size=3,learning_rate=0.0022616698341573717,n_filters=16,n_hidden=1)
plot_model(CNN, show_shapes=True)
history_cnn = CNN.fit(X_train, Y_train, epochs=50,validation_data=(X_valid, Y_valid))
#%%Define the path of CNN
cnn_path = os.path.join('cnn', '001')
#%% Save CNN
model_cnn = rnd_search_cnn.best_estimator_.model
tf.saved_model.save(model_cnn, cnn_path)
#%%Load CNN
model_cnn = tf.saved_model.load(cnn_path)
#%%Random search Wavenet_alter
param_wavenet = {
"n_blocks": [1, 2, 3],
"n_filters": [1,2,4,8,16,32,64,128],
"learning_rate": reciprocal(3e-4, 3e-2),
"kernel_size": [1,2,3,4]
}
def build_wavenet_alter(n_neurons=50,learning_rate=1e-3,input_shape=[n_steps_in,n_features],
              n_blocks=1,n_filters = 64, kernel_size=2):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for i in range(n_blocks):
        for rate in (1, 2, 4, 8) * 2:
            model.add(keras.layers.Conv1D(filters=n_filters,     kernel_size=kernel_size,padding="causal",activation="relu", dilation_rate=rate))
        #model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_steps_out))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
keras_reg_wavenet_alter = keras.wrappers.scikit_learn.KerasRegressor(build_wavenet_alter)
rnd_search_cv_wavenet_alter = RandomizedSearchCV(keras_reg_wavenet_alter, param_wavenet, n_iter=40,scoring = 'neg_mean_squared_error',cv=3)
rnd_search_cv_wavenet_alter.fit(X_train, Y_train, epochs=50,validation_data=(X_valid, Y_valid),batch_size = 32,callbacks=[keras.callbacks.EarlyStopping(patience=10)])
#%%Results of wavenet alter
best_wavenet = rnd_search_cv_wavenet_alter.best_params_
#{'kernel_size': 1,
#'learning_rate': 0.006459905168727454,
#'n_blocks': 1,
#'n_filters': 8}
cvres_wavenet = rnd_search_cv_wavenet_alter.cv_results_
for mean_score, params in zip(cvres_wavenet["mean_test_score"],cvres_wavenet["params"]):
    print(np.sqrt(-mean_score), params)
#0.20235460703383923 {'kernel_size': 2, 'learning_rate': 0.0016510794196647203, 'n_blocks': 3, 'n_filters': 1}
#0.20411360408341436 {'kernel_size': 3, 'learning_rate': 0.023750214374396034, 'n_blocks': 2, 'n_filters': 128}
#0.17634264718450351 {'kernel_size': 3, 'learning_rate': 0.0032635134120246535, 'n_blocks': 1, 'n_filters': 32}
#0.20610768505708502 {'kernel_size': 2, 'learning_rate': 0.005113105945249837, 'n_blocks': 3, 'n_filters': 4}
#0.20225649204904586 {'kernel_size': 2, 'learning_rate': 0.016338741105903515, 'n_blocks': 2, 'n_filters': 64}
#0.20225622266467 {'kernel_size': 1, 'learning_rate': 0.0042537341960849, 'n_blocks': 3, 'n_filters': 4}
#0.20175052244842573 {'kernel_size': 1, 'learning_rate': 0.0022958079841622235, 'n_blocks': 3, 'n_filters': 1}
#0.08637994327428872 {'kernel_size': 1, 'learning_rate': 0.005016173378708309, 'n_blocks': 2, 'n_filters': 16}
#0.21417741313994923 {'kernel_size': 3, 'learning_rate': 0.0016354953740513197, 'n_blocks': 3, 'n_filters': 128}
#0.10240603435787737 {'kernel_size': 1, 'learning_rate': 0.000992819842333296, 'n_blocks': 1, 'n_filters': 64}
#0.14177294413256247 {'kernel_size': 3, 'learning_rate': 0.01451792895684976, 'n_blocks': 1, 'n_filters': 4}
#0.10856691716111906 {'kernel_size': 2, 'learning_rate': 0.006148425269146729, 'n_blocks': 1, 'n_filters': 16}
#0.06519646037657044 {'kernel_size': 1, 'learning_rate': 0.006459905168727454, 'n_blocks': 1, 'n_filters': 8}
#0.09382812806155975 {'kernel_size': 1, 'learning_rate': 0.0010835314012493207, 'n_blocks': 2, 'n_filters': 128}
#0.19309990895788798 {'kernel_size': 3, 'learning_rate': 0.00036472613888787646, 'n_blocks': 1, 'n_filters': 8}
#0.1725313377157677 {'kernel_size': 4, 'learning_rate': 0.00033434880812568, 'n_blocks': 3, 'n_filters': 32}
#0.20191025025573842 {'kernel_size': 4, 'learning_rate': 0.0003900349726363198, 'n_blocks': 1, 'n_filters': 1}
#0.28808621875901624 {'kernel_size': 3, 'learning_rate': 0.023528775622717064, 'n_blocks': 1, 'n_filters': 128}
#0.08210739575794983 {'kernel_size': 1, 'learning_rate': 0.007624931431061324, 'n_blocks': 1, 'n_filters': 8}
#0.19787886731719587 {'kernel_size': 4, 'learning_rate': 0.0017422145482826924, 'n_blocks': 2, 'n_filters': 64}
#0.1355177586745548 {'kernel_size': 1, 'learning_rate': 0.029257513813899804, 'n_blocks': 1, 'n_filters': 16}
#0.20153181900655093 {'kernel_size': 2, 'learning_rate': 0.00033759569456615486, 'n_blocks': 1, 'n_filters': 1}
#0.1664658090167232 {'kernel_size': 2, 'learning_rate': 0.0005588975654351881, 'n_blocks': 3, 'n_filters': 64}
#0.20327310373914953 {'kernel_size': 1, 'learning_rate': 0.014718487359530937, 'n_blocks': 3, 'n_filters': 4}
#0.16344821135167265 {'kernel_size': 2, 'learning_rate': 0.0004482230529998801, #'n_blocks': 1, 'n_filters': 64}
#0.1321057893743768 {'kernel_size': 3, 'learning_rate': 0.008546522117002875, 'n_blocks': 2, 'n_filters': 8}
#0.1990975083148332 {'kernel_size': 2, 'learning_rate': 0.0031484182704411092, 'n_blocks': 3, 'n_filters': 64}
#0.0754989038641941 {'kernel_size': 1, 'learning_rate': 0.0012395767714262276, 'n_blocks': 1, 'n_filters': 16}
#0.19674929007858152 {'kernel_size': 2, 'learning_rate': 0.011341197376687038, 'n_blocks': 2, 'n_filters': 32}
#0.20048710965059544 {'kernel_size': 3, 'learning_rate': 0.0029640604930481354, 'n_blocks': 3, 'n_filters': 2}
#0.18329533412996932 {'kernel_size': 2, 'learning_rate': 0.0003861343426957537, 'n_blocks': 3, 'n_filters': 32}
#0.21775148611892756 {'kernel_size': 2, 'learning_rate': 0.0005596194784503367, 'n_blocks': 2, 'n_filters': 2}
#0.2016014720381246 {'kernel_size': 3, 'learning_rate': 0.003065135127171136, 'n_blocks': 3, 'n_filters': 1}
#0.17922249756218636 {'kernel_size': 3, 'learning_rate': 0.0005163002585725365, 'n_blocks': 2, 'n_filters': 16}
#0.12422032581343254 {'kernel_size': 1, 'learning_rate': 0.01929433726022437, 'n_blocks': 1, 'n_filters': 32}
#0.20103042496505816 {'kernel_size': 2, 'learning_rate': 0.0031186439404980483, 'n_blocks': 3, 'n_filters': 8}
#0.14709700331961978 {'kernel_size': 2, 'learning_rate': 0.003150034997794583, 'n_blocks': 1, 'n_filters': 4}
#0.11532421486023811 {'kernel_size': 1, 'learning_rate': 0.0007072506986603087, 'n_blocks': 3, 'n_filters': 128}
#0.12956682436641406 {'kernel_size': 4, 'learning_rate': 0.00047348440407658516, 'n_blocks': 1, 'n_filters': 8}
#0.20192211923882392 {'kernel_size': 1, 'learning_rate': 0.0008312488842839985, 'n_blocks': 1, 'n_filters': 1}
rnd_search_cv_wavenet_alter.best_params_
#{'kernel_size': 1,
#'learning_rate': 0.0014588181936763893,
#'n_blocks': 2,
#'n_filters': 32}
cvres_wavenet_alter = rnd_search_cv_wavenet_alter.cv_results_
for mean_score, params in zip(cvres_wavenet_alter["mean_test_score"],cvres_wavenet_alter["params"]):
    print(np.sqrt(-mean_score), params)
#0.19971400617052296 {'kernel_size': 1, 'learning_rate': 0.015226901637231876, 'n_blocks': 3, 'n_filters': 2}
#0.20588737682482208 {'kernel_size': 4, 'learning_rate': 0.02636048387558939, 'n_blocks': 3, 'n_filters': 1}
#0.17147655711374457 {'kernel_size': 3, 'learning_rate': 0.003183037187579816, 'n_blocks': 2, 'n_filters': 8}
#0.20010822985760804 {'kernel_size': 2, 'learning_rate': 0.025834157229780174, 'n_blocks': 3, 'n_filters': 2}
#0.1986423815617254 {'kernel_size': 3, 'learning_rate': 0.0003291318968771268, 'n_blocks': 2, 'n_filters': 64}
#0.15263097581928695 {'kernel_size': 4, 'learning_rate': 0.002809670446163642, 'n_blocks': 1, 'n_filters': 2}
#0.19930301398311415 {'kernel_size': 3, 'learning_rate': 0.0006488684282501108, 'n_blocks': 2, 'n_filters': 2}
#0.20107042044081827 {'kernel_size': 4, 'learning_rate': 0.02867791763152855, 'n_blocks': 1, 'n_filters': 16}
#0.1733922541101197 {'kernel_size': 2, 'learning_rate': 0.0010901667606783591, 'n_blocks': 3, 'n_filters': 8}
#0.21061596277213115 {'kernel_size': 2, 'learning_rate': 0.01698350760177379, 'n_blocks': 3, 'n_filters': 16}
#0.2020175425005357 {'kernel_size': 3, 'learning_rate': 0.0004302995272544455, 'n_blocks': 1, 'n_filters': 1}
#0.20068886000199596 {'kernel_size': 4, 'learning_rate': 0.007570601658895483, 'n_blocks': 3, 'n_filters': 8}
#0.20491823129124861 {'kernel_size': 1, 'learning_rate': 0.016055508860121605, 'n_blocks': 3, 'n_filters': 4}
#0.2013567088772558 {'kernel_size': 1, 'learning_rate': 0.00651653693274904, 'n_blocks': 3, 'n_filters': 4}
#0.09732980990251495 {'kernel_size': 1, 'learning_rate': 0.0014588181936763893, 'n_blocks': 2, 'n_filters': 32}
#0.21450004694627 {'kernel_size': 3, 'learning_rate': 0.004994406228071794, 'n_blocks': 2, 'n_filters': 16}
#0.1992668417303434 {'kernel_size': 2, 'learning_rate': 0.006337982570050195, 'n_blocks': 2, 'n_filters': 8}
#0.1984589213241889 {'kernel_size': 1, 'learning_rate': 0.017680296214887627, 'n_blocks': 3, 'n_filters': 1}
#0.1971044283550634 {'kernel_size': 1, 'learning_rate': 0.010968482549639176, 'n_blocks': 1, 'n_filters': 2}
#0.1395097650531935 {'kernel_size': 2, 'learning_rate': 0.0011211276367843453, 'n_blocks': 1, 'n_filters': 128}
#0.20282863059141973 {'kernel_size': 2, 'learning_rate': 0.019203652021162074, 'n_blocks': 3, 'n_filters': 32}
#0.1776661344202109 {'kernel_size': 4, 'learning_rate': 0.0003200723580124724, 'n_blocks': 1, 'n_filters': 64}
#0.20235398674146537 {'kernel_size': 4, 'learning_rate': 0.00877249480547499, 'n_blocks': 2, 'n_filters': 128}
#0.20180895322967057 {'kernel_size': 2, 'learning_rate': 0.000715557079529725, 'n_blocks': 3, 'n_filters': 1}
#0.2057301896747153 {'kernel_size': 4, 'learning_rate': 0.012375347600987718, 'n_blocks': 1, 'n_filters': 16}
#0.20215521059456673 {'kernel_size': 1, 'learning_rate': 0.006131532943149156, 'n_blocks': 2, 'n_filters': 1}
#0.09947977301585881 {'kernel_size': 1, 'learning_rate': 0.009784525155890554, 'n_blocks': 1, 'n_filters': 64}
#0.15556111648390564 {'kernel_size': 2, 'learning_rate': 0.0005415214243906932, 'n_blocks': 3, 'n_filters': 16}
#0.16059749013247823 {'kernel_size': 4, 'learning_rate': 0.013628588580571435, 'n_blocks': 1, 'n_filters': 4}
#0.19794398097641563 {'kernel_size': 2, 'learning_rate': 0.007973177583247334, 'n_blocks': 2, 'n_filters': 64}
#0.12260062998068516 {'kernel_size': 2, 'learning_rate': 0.0003173544937230259, 'n_blocks': 1, 'n_filters': 16}
#0.2019033352071174 {'kernel_size': 3, 'learning_rate': 0.0003832569785249008, 'n_blocks': 2, 'n_filters': 1}
#0.2015680889322474 {'kernel_size': 2, 'learning_rate': 0.0022083577599571855, 'n_blocks': 2, 'n_filters': 1}
#0.18062696645946874 {'kernel_size': 4, 'learning_rate': 0.0010311795816122479, 'n_blocks': 2, 'n_filters': 16}
#0.13380393572272922 {'kernel_size': 2, 'learning_rate': 0.0021991266012631803, 'n_blocks': 1, 'n_filters': 128}
#0.1528385906502109 {'kernel_size': 2, 'learning_rate': 0.0012733223685982178, 'n_blocks': 1, 'n_filters': 128}
#0.19795729198422082 {'kernel_size': 1, 'learning_rate': 0.0235079751747334, 'n_blocks': 2, 'n_filters': 32}
#0.2016549602572939 {'kernel_size': 2, 'learning_rate': 0.00999714368833112, 'n_blocks': 3, 'n_filters': 2}
#0.19999341490905537 {'kernel_size': 3, 'learning_rate': 0.017920265266791583, 'n_blocks': 3, 'n_filters': 1}
#0.20038940159878207 {'kernel_size': 4, 'learning_rate': 0.024480085325094693, 'n_blocks': 1, 'n_filters': 16}
#%%Define the path of WaveNet
wavenet_path = os.path.join('WaveNet', '002')
#%% Save WaveNet
model_wavenet_alter = rnd_search_cv_wavenet_alter.best_estimator_.model
tf.saved_model.save(model_wavenet_alter, wavenet_path)
#%%Load WaveNet
model_wavenet_alter = tf.saved_model.load(wavenet_path)
#%%
mse_linear_train = mean_squared_error(Y_train,linear2.predict(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])))
mse_linear_valid = mean_squared_error(Y_valid,linear2.predict(X_valid.reshape(X_valid.shape[0], X_valid.shape[1]*X_valid.shape[2])))
print('the mse of linear regression is {0} {1} {2}'.format(mse_linear, mse_linear_train, mse_linear_valid))
#the mse of linear regression is 0.002201485885414259 0.0002386875309150674 0.002883088439729552
mae_linear_train = mean_absolute_error(Y_train,linear2.predict(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])))
mae_linear_valid = mean_absolute_error(Y_valid,linear2.predict(X_valid.reshape(X_valid.shape[0], X_valid.shape[1]*X_valid.shape[2])))
mae_linear = mean_absolute_error(Y_test,linear2.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])))
print('the mae of linear regression is {0} {1} {2}'.format(mae_linear, mae_linear_train, mae_linear_valid))
#the mae of linear regression is 0.037201667427589534 0.0111178498822272 0.038246402212894735



mse_ann_train=mean_squared_error(Y_train,ann.predict(X_train))
mse_ann_valid=mean_squared_error(Y_valid,ann.predict(X_valid))
mse_ann_test=mean_squared_error(Y_test,ann.predict(X_test))
print('the mse of ANN is {0} {1} {2}'.format(mse_ann_test, mse_ann_train, mse_ann_valid))
#the mse of ANN is 0.0017354279248674228 0.00046761570091655966 0.003913055540831749
mae_ann_train=mean_absolute_error(Y_train,ann.predict(X_train))
mae_ann_valid=mean_absolute_error(Y_valid,ann.predict(X_valid))
mae_ann_test=mean_absolute_error(Y_test,ann.predict(X_test))
print('the mae of ANN is {0} {1} {2}'.format(mae_ann_test, mae_ann_train, mae_ann_valid))
#the mae of ANN is 0.03499685428963598 0.015582769387284767 0.04510676395174744


predictions_lstm=model_lstm(tf.cast(X_test,dtype=tf.float32),training= False)
mse_lstm = mean_squared_error(Y_test, predictions_lstm)
mse_lstm_train = mean_squared_error(Y_train,model_lstm(tf.cast(X_train,dtype=tf.float32),training= False))
mse_lstm_valid = mean_squared_error(Y_valid,model_lstm(tf.cast(X_valid,dtype=tf.float32),training= False))
print('the mse of LSTM is {0} {1} {2}'.format(mse_lstm,mse_lstm_train,mse_lstm_valid))
#the mse of LSTM is 0.0012379190827367195 0.0005582953650761166 0.0032913907791190028

mae_lstm_test = mean_absolute_error(Y_test, model_lstm(tf.cast(X_test,dtype=tf.float32),training= False))
mae_lstm_train = mean_absolute_error(Y_train,model_lstm(tf.cast(X_train,dtype=tf.float32),training= False))
mae_lstm_valid = mean_absolute_error(Y_valid,model_lstm(tf.cast(X_valid,dtype=tf.float32),training= False))
print('the mae of LSTM is {0} {1} {2}'.format(mae_lstm_test,mae_lstm_train,mae_lstm_valid))
#the mae of LSTM is 0.027064870138872394 0.016271803353243554 0.03673457810467793




model_gru = rnd_search_gru.best_estimator_.model
mse_gru_train = mean_squared_error(Y_train,model_gru.predict(X_train))
mse_gru_valid = mean_squared_error(Y_valid,model_gru.predict(X_valid))
predictions_gru = model_gru.predict(X_test)
mse_gru = mean_squared_error(Y_test, predictions_gru)
print('the mse of GRU is {0} {1} {2}'.format(mse_gru,mse_gru_train,mse_gru_valid))
#the mse of GRU is 0.0006278977329382915 0.0004994875812914647 0.0010814193526827213
mae_gru_train = mean_absolute_error(Y_train,model_gru.predict(X_train))
mae_gru_valid = mean_absolute_error(Y_valid,model_gru.predict(X_valid))
mae_gru = mean_absolute_error(Y_test, model_gru.predict(X_test))
print('the mae of GRU is {0} {1} {2}'.format(mae_gru,mae_gru_train,mae_gru_valid))
#the mae of GRU is 0.01883551352946973 0.016618429158539085 0.022119827484296375


model_cnn = rnd_search_cnn.best_estimator_.model
mse_cnn_train = mean_squared_error(Y_train,model_cnn.predict(X_train))
mse_cnn_valid = mean_squared_error(Y_valid,model_cnn.predict(X_valid))
predictions_cnn = model_cnn.predict(X_test)
mse_cnn = mean_squared_error(Y_test, predictions_cnn)
print('the mse of CNN is {0} {1} {2}'.format(mse_cnn,mse_cnn_train,mse_cnn_valid))
#the mse of CNN is 0.0010615718719568005 0.0005528686339129492 0.0019073838110821705

mae_cnn_train = mean_absolute_error(Y_train,model_cnn.predict(X_train))
mae_cnn_valid = mean_absolute_error(Y_valid,model_cnn.predict(X_valid))
predictions_cnn = model_cnn.predict(X_test)
mae_cnn = mean_absolute_error(Y_test, predictions_cnn)
print('the mae of CNN is {0} {1} {2}'.format(mae_cnn,mae_cnn_train,mae_cnn_valid))
#the mse of CNN is 0.026568108924616425 0.01619821738033063 0.031800133054219165


model_wavenet_alter = rnd_search_cv_wavenet_alter.best_estimator_.model
mse_wavenet_alter_train = mean_squared_error(Y_train,model_wavenet_alter.predict(X_train))
mse_wavenet_alter_valid = mean_squared_error(Y_valid,model_wavenet_alter.predict(X_valid))
predictions_wavenet_alter = model_wavenet_alter.predict(X_test)
mse_wavenet_alter = mean_squared_error(Y_test, predictions_wavenet_alter)
print('the mse of WaveNet_alter is {0} {1} {2}'.format(mse_wavenet_alter,mse_wavenet_alter_train,mse_wavenet_alter_valid))
#the mse of WaveNet_alter is 0.0008568894022378818 0.0005506116345980296 0.0014721166440331956

mae_wavenet_alter_train = mean_absolute_error(Y_train,model_wavenet_alter.predict(X_train))
mae_wavenet_alter_valid = mean_absolute_error(Y_valid,model_wavenet_alter.predict(X_valid))
predictions_wavenet_alter = model_wavenet_alter.predict(X_test)
mae_wavenet_alter = mean_absolute_error(Y_test, predictions_wavenet_alter)
print('the mae of WaveNet_alter is {0} {1} {2}'.format(mae_wavenet_alter,mae_wavenet_alter_train,mae_wavenet_alter_valid))
   
