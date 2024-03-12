import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = {
    'Year': list(range(1980, 2021)),
    'CO2': [
        7317185638, 7654290821, 8002106455, 8378782780, 8764711611, 9188967638, 9644034163, 10130101461, 10655183128,
        11231587676, 11848166774, 12504907667, 13203664617, 13926558876, 14690800843, 15502261630, 16381088594,
        17296137907, 18229687165, 19224920237, 20254557996, 21290198710, 22337794361, 23429667371, 24575212605,
        25785358272, 27072507483, 28462760974, 30010752013, 31730872573, 33430899968, 35248071567, 37231830380,
        39229708958, 41437242850, 43713650045, 46091097909, 48557863281, 2458175.90, 2423951, 2200836.30
    ]
}

df = pd.DataFrame(data)

# Clean CO2 column and convert to numerical
df['CO2'] = df['CO2'].astype(str).str.replace(',', '').astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['CO2'].values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Define a function to create X and y datasets
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Reshape into X=t, t+1, t+2, t+3 and y=t+4
time_step = 1
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, time_step)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=1, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transformation to get original values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
import math
from sklearn.metrics import mean_squared_error

print("Train RMSE:", math.sqrt(mean_squared_error(y_train_inv, train_predict)))
print("Test RMSE:", math.sqrt(mean_squared_error(y_test_inv, test_predict)))

# Last available year
last_year = df['Year'].iloc[-1]

# Reshape the last data point for prediction
last_data_point = scaled_data[-time_step:, 0].reshape(1, 1, time_step)

# Predict CO2 emissions for the next 10 years
future_predictions = []
for i in range(10):
    prediction = model.predict(last_data_point)
    future_predictions.append(prediction[0, 0])
    last_data_point = np.append(last_data_point[:, :, 1:], prediction).reshape(1, 1, time_step)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Display the predicted CO2 emissions for the next 10 years
for i in range(1, 11):
    print("Predicted CO2 emissions for year {}: {:.2f}".format(last_year + i, future_predictions[i-1][0]))
