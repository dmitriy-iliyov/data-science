from lstm.preparer import prepare_data
from model import KerasLSTM


reviews_count = 100000
max_length = 100
train_data, train_answers, test_data, test_answers, word_count = prepare_data(reviews_count, 0.9, max_length)

print(f'Vocabulary length: {word_count}')

lstm = KerasLSTM(word_count, max_length)
lstm.summary()

lstm.fit(train_data, train_answers)

count = 20

predicting_data = test_data[:count].copy()
predicting_answers = test_answers[:count].copy().tolist()
test_data = test_data[count:]
test_answers = test_answers[count:]

print(f"Test data count: {len(test_data)}, {len(test_answers)}")
lstm.evaluate(test_data, test_answers)

print(predicting_answers[0])
for i, data in enumerate(predicting_data):
    data = data.reshape(1, -1)
    predicted_vec = lstm.predict(data)[0].tolist()
    max_val_in_vec = max(predicted_vec)
    predicted_val_index = predicted_vec.index(max_val_in_vec)
    predicterd_val = predicted_val_index + 1
    print(f"real stars: {(predicting_answers[i].index(0)) + 1}; predicted: {predicterd_val}")