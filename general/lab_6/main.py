import numpy as np
from sklearn.preprocessing import MinMaxScaler

from lab_1 import parser
from keras_lstm import KerasLSTM


scaler = MinMaxScaler(feature_range=(0, 1))


def prepare_data(currency):
    df = parser.coin_parsing(currency, 90)
    prices = df['price']

    prices_scaled = scaler.fit_transform(prices.values.reshape(-1, 1))

    print(f"Prices scaled shape: {prices_scaled.shape}")

    time_step = 24
    features = 1
    train_data = []
    train_answ = []

    for i in range(len(prices_scaled) - time_step):
        train_data.append(prices_scaled[i:i + time_step])
        train_answ.append(prices_scaled[i + time_step])

    train_data = np.array(train_data, dtype=np.float32)
    train_answ = np.array(train_answ, dtype=np.float32)

    k = int(train_data.shape[0] * 0.8)
    train_data, test_data = train_data[:k], train_data[k:]
    train_answ, test_answ = train_answ[:k], train_answ[k:]

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], features)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], features)

    return train_data, train_answ, test_data, test_answ


train_data, train_answers, test_data, test_answers = prepare_data("bitcoin")

lstm = KerasLSTM()
lstm.summary()
lstm.fit(train_data, train_answers)
lstm.evaluate(test_data, test_answers)

last_48_hour = parser.coin_parsing('bitcoin', 2)
sequence = last_48_hour['price'][:24]
sequence_scaled = scaler.transform(np.array(sequence).reshape(-1, 1)).reshape(1, 24, 1)

prediction = lstm.predict(sequence_scaled)
predicted_price = scaler.inverse_transform(prediction)
print(f"Predicted price: {predicted_price[0][0]}")
print(f"Actual price: {last_48_hour['price'][24]}")
