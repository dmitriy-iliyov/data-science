from lab_2_fce.lab_2_keras.models import keras_fnn, keras_cnn, keras_enn
import lab_2_fce.lab_2.tools.data_processing as dp
from lab_2_fce.lab_2_keras.tools.network_compare import NetworkComparor


train_data, train_answers, test_data, test_answers = dp.prepared_data(-1, 1, 100)

fnn_t1 = keras_fnn.KerasFNN(2, 1, 10)
fnn_t2 = keras_fnn.KerasFNN(2, 1, 20)

fnn_comparator = NetworkComparor(fnn_t1, fnn_t2)
fnn_comparator.fit([train_data, train_answers])
fnn_comparator.predict([test_data, test_answers])
fnn_comparator.print_plots()

cnn_t1 = keras_cnn.KerasCNN(2, 1, 20)
cnn_t2 = keras_cnn.KerasCNN(2, 2, 10)

cnn_comparator = NetworkComparor(cnn_t1, cnn_t2)
cnn_comparator.fit([train_data, train_answers])
cnn_comparator.predict([test_data, test_answers])
cnn_comparator.print_plots()

enn_t1 = keras_enn.KerasENN(2, 1, 15)
enn_t2 = keras_enn.KerasENN(2, 3, 5)

enn_comparator = NetworkComparor(enn_t1, enn_t2)
enn_comparator.fit([train_data, train_answers])
enn_comparator.predict([test_data, test_answers])
enn_comparator.print_plots()
