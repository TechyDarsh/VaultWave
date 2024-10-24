import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = pd.read_csv(r"C:\Users\darsh\OneDrive\Desktop\stock_datasets\nifty_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


X, y = prepare_data(scaled_data, n_steps)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 50
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

predicted_dates = data.index[-len(predictions):]
predicted_df = pd.DataFrame(data={'Predicted': predictions.flatten()}, index=predicted_dates)

def predict_future(model, last_sequence, n_steps, n_future_steps):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future_steps):
        prediction = model.predict(current_sequence.reshape(1, n_steps, 1))
        future_predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction)
    
    return np.array(future_predictions)

last_sequence = scaled_data[-n_steps:]
n_future_steps = 2
future_predictions = predict_future(model, last_sequence, n_steps, n_future_steps)
future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

last_date = data.index[-1]
future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, n_future_steps + 1)]
future_df = pd.DataFrame(data={'Predicted': future_predictions.flatten()}, index=future_dates)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], color='blue', label='Actual Prices')
plt.plot(predicted_df.index, predicted_df['Predicted'], color='red', linestyle='--', label='Test Set Predictions')
plt.plot(future_df.index, future_df['Predicted'], color='green', linestyle='--', label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

future_min = np.min(future_predictions)
future_max = np.max(future_predictions)
print(f"Future predicted price range: {future_min:.2f} - {future_max:.2f}")
