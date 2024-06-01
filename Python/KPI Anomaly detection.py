#!/usr/bin/env python
# coding: utf-8

# # KPI Anomaly detection

# ## Data exploration and cleaning

# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import *

import tensorflow as tf
from tensorflow.keras import layers, Model
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
import joblib


# In[3]:


# Read dataframe
dlpdcp_volume = pd.read_csv('C:/Users/ASUS/OneDrive/Escritorio/Datos/Maestria/TFM/Reference data/KPIs_calcutaled/dlpdcp_volume.csv')
dlpdcp_volume


# In[4]:


# Graph dlpdcp_volume
plt.figure(figsize=(15, 6))
neIds = dlpdcp_volume['neId'].unique()
for neId in neIds:
    df_neId = dlpdcp_volume[dlpdcp_volume['neId'] == neId]
    df_neId = df_neId.sort_values(by='startTime')
    n_rows = len(df_neId)
    n_rows_sel = int(n_rows * 0.05)
    df_neId_sample = df_neId.iloc[:n_rows_sel]
    df_neId_dt = df_neId_sample.copy()
    df_neId_dt['startTime'] = pd.to_datetime(df_neId_dt['startTime'], unit='ms')
    plt.plot(df_neId_dt['startTime'], df_neId_dt['kpiValue'], label=neId)

plt.title('dlpdcp volume trend')
plt.xlabel('startTime')
plt.ylabel('kpiValue')

plt.legend()
plt.show()


# In[5]:


# Split dataframe by neId
for neId, group_df in dlpdcp_volume.groupby('neId'):
    df_name = neId.lower()[0:3] + '_dlpdcp_volume'
    globals()[df_name] = group_df

dfs = {
'bcn_dlpdcp_volume': bcn_dlpdcp_volume,
'tar_dlpdcp_volume': tar_dlpdcp_volume,
'mad_dlpdcp_volume': mad_dlpdcp_volume,
'tol_dlpdcp_volume': tol_dlpdcp_volume,
'sal_dlpdcp_volume': sal_dlpdcp_volume,
'val_dlpdcp_volume': val_dlpdcp_volume,
'bil_dlpdcp_volume': bil_dlpdcp_volume,
'sev_dlpdcp_volume': sev_dlpdcp_volume,
'snt_dlpdcp_volume': snt_dlpdcp_volume
}

for df_name, df in dfs.items():
    print(df_name)
    print(df.info())
    print() 


# In[6]:


# Delete non required attributes
def clean_attributes(df):
    df_cln = df.copy()
    df_cln.drop(["neId","kpiName"], axis = 1, inplace = True)
    df_cln = df_cln.set_index('startTime')
    df_cln.index = pd.to_datetime(df_cln.index, unit='ms')
    return df_cln

dfs_cln = {}
for key, df in dfs.items():
    dfs_cln[key] = clean_attributes(df)


# In[7]:


# Check descriptive statistics
for df_name, df in dfs_cln.items():
    print(df_name)
    print(df.describe())
    print() 
    print()     


# In[8]:


# Graph histograms
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, (df_name, df) in enumerate(dfs_cln.items()):
    row = i // 3
    column = i % 3
    
    axes[row, column].hist(df['kpiValue'], bins=30, color='skyblue', edgecolor='black')
    axes[row, column].set_title(f'{df_name} Histogram')
    axes[row, column].set_xlabel('kpiValue')
    axes[row, column].set_ylabel('Freq')
    axes[row, column].grid(True)

plt.tight_layout()
plt.show()


# In[9]:


# Graph box plots
fig, axes = plt.subplots(len(dfs), 1, figsize=(15, 2*len(dfs_cln)))

for i, (df_name, df) in enumerate(dfs_cln.items()):
    sns.boxplot(ax=axes[i], x=df['kpiValue'], data=df, palette="Set2")
    axes[i].set_title(f'{df_name} Boxplot', fontsize=11)
    axes[i].set_xlabel('kpiValue')
    axes[i].set_ylabel('')
    axes[i].grid(True)

plt.tight_layout()
plt.show()


# In[10]:


# Calculate missing values percentage
for df_name, df in dfs_cln.items():
    total = len(df)
    nulls_count = df['kpiValue'].isnull().sum()
    perc = 100*round(nulls_count/total,4)
    print(f"{df_name} - nulls: {nulls_count} ({perc}%)")


# In[11]:


# Interpolate nulls
def interpolate_nulls(df):
    df_no_nulls = df.copy()
    df_no_nulls.interpolate(method='time', inplace=True)
    return df_no_nulls

dfs_no_nulls = {}
for df_name, df in dfs_cln.items():
    df_no_nulls = interpolate_nulls(df)
    dfs_no_nulls[df_name] = df_no_nulls


# In[12]:


# Validate nulls
for df_name, df in dfs_no_nulls.items():
    val_null = df.isnull().values.any()
    print(df_name, '- nulls: ',val_null)


# ## Models evaluation (dataframe: bcn_dlpdcp_volume)

# In[13]:


# Dataset preparation
bcn_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['bcn_dlpdcp_volume'])


# In[14]:


# Divide datasets
def split_datasets(df):
    train_data, no_train_data = train_test_split(df, test_size=0.3, shuffle=False)
    validation_data, test_data = train_test_split(no_train_data, test_size=0.5, shuffle=False) 
    print("Tranning data records:", len(train_data))
    print("Validation data records:", len(validation_data))
    print("Test data records:", len(test_data))
    return train_data, validation_data, test_data

train_data, validation_data, test_data = split_datasets(bcn_dlpdcp_volume_cln.copy())


# In[15]:


# Graph datasets trend
datasets = [(train_data, 'Training'), (validation_data, 'Validation'), (test_data, 'Test')]

for dataset, dataset_name in datasets:
    fig = go.Figure()  
    fig.add_scatter(x=dataset.index, y=dataset['kpiValue'], mode='lines', name=dataset_name)
    fig.update_layout(
        xaxis_title='startTime',
        yaxis_title='kpiValue',
        title=f'Data bcn dlpdcp volume ({dataset_name})',
        hovermode='x'
    )
    fig.show()


# ### Model 1: IsolationForest

# In[16]:


# Train model 
def train_if(train_data):
    #cont = 1/len(train_data)
    cont = 0.05
    model = IsolationForest(contamination= cont, random_state=327)
    train_data_sh = train_data['kpiValue'].values.reshape(-1,1)
    model.fit(train_data_sh)
    return model


# In[17]:


# Predict values
def predict_if(validation_data, model):
    predictions = model.predict(validation_data['kpiValue'].values.reshape(-1,1))
    return predictions


# In[18]:


# Get anomalies
def anomalies_if(validation_data, predictions):
    anomalies_indices = [i for i, pred in enumerate(predictions) if pred == -1]
    anomalies_values = [validation_data.iloc[i]['kpiValue'] for i in anomalies_indices]
    print("Anomalies detected: ", len(anomalies_indices))
    return anomalies_indices, anomalies_values


# In[19]:


# Graph predictions
def graph_if(validation_data, anomalies_indices, anomalies_values):
    mod = 'Isolation Forest'
    fig = go.Figure()
    fig.add_scatter(x=validation_data.index, y=validation_data['kpiValue'], mode='lines', name='original')
    fig.add_scatter(x=validation_data.index[anomalies_indices], y=anomalies_values, mode='markers', marker=dict(color='red'), name='anomalies')
    fig.update_layout(
        xaxis_title='startTime',
        yaxis_title='kpiValue',
        title='Anomalies '+mod,
        hovermode='x')
    fig.show()


# In[20]:


# Execute and validate model
model_if = train_if(train_data)
predictions_if = predict_if(validation_data, model_if)
anomalies_indices_if, anomalies_values_if = anomalies_if(validation_data, predictions_if)
graph_if(validation_data, anomalies_indices_if, anomalies_values_if)


# ### Model 2: Local Outlier Factor

# In[21]:


# Train model 
def train_lof(train_data):
    cont = 0.05
    model = LocalOutlierFactor(contamination=cont, novelty=True, n_neighbors=20)
    train_data_sh = train_data['kpiValue'].values.reshape(-1,1)
    model.fit(train_data_sh)
    return model


# In[22]:


# Predict values
def predict_lof(validation_data, model):
    predictions = model.predict(validation_data['kpiValue'].values.reshape(-1,1))
    return predictions


# In[23]:


# Get anomalies
def anomalies_lof(validation_data, predictions):
    anomalies_indices = [i for i, pred in enumerate(predictions) if pred == -1]
    anomalies_values = [validation_data.iloc[i]['kpiValue'] for i in anomalies_indices]
    print("Anomalies detected: ", len(anomalies_indices))
    return anomalies_indices, anomalies_values


# In[24]:


# Graph predictions
def graph_lof(validation_data, anomalies_indices, anomalies_values):
    mod = 'Local Outlier Factor'
    fig = go.Figure()
    fig.add_scatter(x=validation_data.index, y=validation_data['kpiValue'], mode='lines', name='original')
    fig.add_scatter(x=validation_data.index[anomalies_indices], y=anomalies_values, mode='markers', marker=dict(color='red'), name='anomalies')
    fig.update_layout(
        xaxis_title='startTime',
        yaxis_title='kpiValue',
        title='Anomalies '+mod,
        hovermode='x')
    fig.show()


# In[25]:


# Execute and validate model
model_lof = train_lof(train_data)
predictions_lof = predict_lof(validation_data, model_lof)
anomalies_indices_lof, anomalies_values_lof = anomalies_lof(validation_data, predictions_lof)
graph_lof(validation_data, anomalies_indices_lof, anomalies_values_lof)


# ### Model 3: Autoencoder LSTM

# In[26]:


# Normalize data 
def normalize_datasets(train_data, validation_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data_norm = (train_data - mean) / std
    validation_data_norm = (validation_data - mean) / std
    print("Mean: ", mean)
    print("Standard Deviation: ", std)
    return train_data_norm, validation_data_norm


# In[27]:


# Build model
def build_aec_lstm(train_data_norm):
    ext_lstm_units = 64
    int_lstm_units = 32
    input_shape = (train_data_norm.shape[1], 1)
    input_layer = Input(shape=input_shape)
    encoder = LSTM(ext_lstm_units, activation='relu', return_sequences=True)(input_layer)
    encoder_1 = LSTM(int_lstm_units, activation='relu')(encoder)
    repeat = RepeatVector(input_shape[0])(encoder_1)
    decoder = LSTM(int_lstm_units, activation='relu', return_sequences=True)(repeat)
    decoder_1 = LSTM(ext_lstm_units, activation='relu', return_sequences=True)(decoder)
    output_layer = TimeDistributed(Dense(input_shape[1]))(decoder_1)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


# In[28]:


# Train model 
def train_aec_lstm(train_data_norm, model):
    history = model.fit(train_data_norm, train_data_norm, epochs=3, batch_size=128, validation_split=0.1)
    return history


# In[29]:


# Predict values
def predict_aec_lstm(validation_data_norm, model):
    predictions = model.predict(validation_data_norm['kpiValue'].values.reshape(-1,1))
    return predictions


# In[30]:


# Get anomalies
def anomalies_aec_lstm(validation_data_norm, validation_data, predictions):
    mse = np.mean(np.power(validation_data_norm - predictions.reshape(validation_data_norm.shape[0],1), 2), axis=1)
    print("Mean Square Error = ", mse)
    threshold  = 0.05
#    threshold = np.mean(mse) + 3*np.std(mse)
    print("Threshold = ", np.round(threshold, 2))
    anomalies = np.where(mse > np.round(threshold, 2))[0]
    print("Anomalies detected: ", len(anomalies))
    anomalies_indices = list(anomalies)
    anomalies_values = [validation_data.iloc[i]['kpiValue'] for i in anomalies_indices]
    return anomalies_indices, anomalies_values


# In[31]:


# Graph predictions
def graph_aec_lstm(validation_data, anomalies_indices, anomalies_values):
    mod = 'Autoencoder LSTM'
    fig = go.Figure()
    fig.add_scatter(x=validation_data.index, y=validation_data['kpiValue'], mode='lines', name='original')
    fig.add_scatter(x=validation_data.index[anomalies_indices], y=anomalies_values, mode='markers', marker=dict(color='red'), name='anomalies')  
    fig.update_layout(
        xaxis_title='startTime',
        yaxis_title='kpiValue',
        title='Anomalies '+mod,
        hovermode='x')
    fig.show()


# In[32]:


# Execute and validate model
np.random.seed(327)
tf.random.set_seed(327)
keras.utils.set_random_seed(327)

train_data_norm, validation_data_norm = normalize_datasets(train_data, validation_data)
model_aec_lstm = build_aec_lstm(train_data_norm)
history_aec_lstm = train_aec_lstm(train_data_norm, model_aec_lstm)
predictions_aec_lstm = predict_aec_lstm(validation_data_norm, model_aec_lstm)
anomalies_indices_aec_lstm, anomalies_values_aec_lstm = anomalies_aec_lstm(validation_data_norm, validation_data, predictions_aec_lstm)
graph_aec_lstm(validation_data, anomalies_indices_aec_lstm, anomalies_values_aec_lstm)


# ### Model 4: Autoencoder Convolutional

# In[33]:


# Create sequences
def create_seq(dataset, time_steps): 
    output = []
    values = dataset.values
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    seqs = np.stack(output)
    print("Sequences shape: ", seqs.shape)
    return seqs


# In[34]:


# Build model
def build_aec_conv(seq_train):
    model = keras.Sequential([
            layers.Input(shape=(seq_train.shape[1], seq_train.shape[2])),
            layers.Conv1D(filters=32,kernel_size=7,padding="same",strides=2,activation="relu",),
            layers.Dropout(rate=0.2),
            layers.Conv1D(filters=16,kernel_size=7,padding="same",strides=2,activation="relu",),
            layers.Conv1DTranspose(filters=16,kernel_size=7,padding="same",strides=2,activation="relu",),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(filters=32,kernel_size=7,padding="same",strides=2,activation="relu",),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    return model


# In[35]:


# Train model 
def train_aec_conv(seq_train, model):
    model.fit(seq_train,seq_train,epochs=30,batch_size=128,validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],)
    return model


# In[36]:


# Predict values
def predict_aec_conv(seq_data, model):
    predictions = model.predict(seq_data)
    return predictions


# In[37]:


# Get anomalies
def anomalies_aec_conv(seq_data_train, seq_data_val, predictions_train, predictions_val, validation_data):
    mae_loss_train = np.mean(np.abs(predictions_train - seq_data_train), axis=1).reshape((-1))
    mae_loss_val = np.mean(np.abs(predictions_val - seq_data_val), axis=1).reshape((-1))  
    threshold = np.max(mae_loss_train)
    # threshold = 0.1
    print("Reconstruction error threshold: ", threshold)
    anomalies = mae_loss_val > threshold
    print("Anomalies detected: ", np.sum(anomalies))
    anomalies_indices = [i for i, valor in enumerate(anomalies) if valor == True]
    anomalies_values = [validation_data.iloc[i]['kpiValue'] for i in anomalies_indices]
    return anomalies_indices, anomalies_values  


# In[38]:


# Graph predictions
def graph_aec_conv(validation_data, anomalies_indices, anomalies_values):
    mod = 'Autoencoder Convolutional'
    fig = go.Figure()
    fig.add_scatter(x=validation_data.index, y=validation_data['kpiValue'], mode='lines', name='original')
    fig.add_scatter(x=validation_data.index[anomalies_indices], y=anomalies_values, mode='markers', marker=dict(color='red'), name='anomalies')  
    fig.update_layout(
        xaxis_title='startTime',
        yaxis_title='kpiValue',
        title='Anomalies '+mod,
        hovermode='x')
    fig.show()


# In[39]:


# Execute and validate model
np.random.seed(327)
tf.random.set_seed(327)
keras.utils.set_random_seed(327)
time_steps = 96

train_data_norm, validation_data_norm = normalize_datasets(train_data, validation_data)
seq_train = create_seq(train_data_norm, time_steps)
model_aec_conv = build_aec_conv(seq_train)
model_aec_conv = train_aec_conv(seq_train, model_aec_conv)
seq_val = create_seq(validation_data_norm, time_steps)
predictions_train = predict_aec_conv(seq_train, model_aec_conv)
predictions_val = predict_aec_conv(seq_val, model_aec_conv)
anomalies_indices_aec_conv, anomalies_values_aec_conv = anomalies_aec_conv(seq_train, seq_val, predictions_train, predictions_val, validation_data)
graph_aec_conv(validation_data, anomalies_indices_aec_conv, anomalies_values_aec_conv)


# ## Apply model

# In[40]:


# Model function (Isolation Forest)
def apply_ad_model(train_data, validation_data): 
    model_if = train_if(train_data)
    predictions_if = predict_if(validation_data, model_if)
    anomalies_indices_if, anomalies_values_if = anomalies_if(validation_data, predictions_if)
    graph_if(validation_data, anomalies_indices_if, anomalies_values_if)
    return model_if


# In[41]:


# bcn_dlpdcp_volume
bcn_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['bcn_dlpdcp_volume'])
train_data_bcn, validation_data_bcn, test_data_bcn = split_datasets(bcn_dlpdcp_volume_cln.copy())
model_bcn = apply_ad_model(train_data_bcn, validation_data_bcn)


# In[42]:


# tar_dlpdcp_volume
tar_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['tar_dlpdcp_volume'])
train_data_tar, validation_data_tar, test_data_tar = split_datasets(tar_dlpdcp_volume_cln.copy())
model_tar = apply_ad_model(train_data_tar, validation_data_tar)


# In[43]:


# mad_dlpdcp_volume
mad_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['mad_dlpdcp_volume'])
train_data_mad, validation_data_mad, test_data_mad = split_datasets(mad_dlpdcp_volume_cln.copy())
model_mad = apply_ad_model(train_data_mad, validation_data_mad)


# In[44]:


# tol_dlpdcp_volume
tol_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['tol_dlpdcp_volume'])
train_data_tol, validation_data_tol, test_data_tol = split_datasets(tol_dlpdcp_volume_cln.copy())
model_tol = apply_ad_model(train_data_tol, validation_data_tol)


# In[45]:


# sal_dlpdcp_volume
sal_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['sal_dlpdcp_volume'])
train_data_sal, validation_data_sal, test_data_sal = split_datasets(sal_dlpdcp_volume_cln.copy())
model_sal = apply_ad_model(train_data_sal, validation_data_sal)


# In[46]:


# val_dlpdcp_volume
val_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['val_dlpdcp_volume'])
train_data_val, validation_data_val, test_data_val = split_datasets(val_dlpdcp_volume_cln.copy())
model_val = apply_ad_model(train_data_val, validation_data_val)


# In[47]:


# bil_dlpdcp_volume
bil_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['bil_dlpdcp_volume'])
train_data_bil, validation_data_bil, test_data_bil = split_datasets(bil_dlpdcp_volume_cln.copy())
model_bil = apply_ad_model(train_data_bil, validation_data_bil)


# In[48]:


# sev_dlpdcp_volume
sev_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['sev_dlpdcp_volume'])
train_data_sev, validation_data_sev, test_data_sev = split_datasets(sev_dlpdcp_volume_cln.copy())
model_sev = apply_ad_model(train_data_sev, validation_data_sev)


# In[49]:


# snt_dlpdcp_volume
snt_dlpdcp_volume_cln = pd.DataFrame(dfs_no_nulls['snt_dlpdcp_volume'])
train_data_snt, validation_data_snt, test_data_snt = split_datasets(snt_dlpdcp_volume_cln.copy())
model_snt = apply_ad_model(train_data_snt, validation_data_snt)


# ## Save models

# In[50]:


# Dump models
models = {
    'model_bcn': model_bcn,
    'model_tar': model_tar,
    'model_mad': model_mad,
    'model_tol': model_tol,
    'model_sal': model_sal,
    'model_val': model_val,
    'model_bil': model_bil,
    'model_sev': model_sev,
    'model_snt': model_snt
}

for model_name, model in models.items():
    file_name = f'{model_name}_anomaly_detection.pkl'
    joblib.dump(model, file_name)


# In[ ]:




