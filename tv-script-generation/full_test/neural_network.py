import tensorflow as tf

class NeuralNetwork():
    def get_inputs(self):
        p_input = tf.placeholder(tf.int32, [None, None], name="input")
        p_targets = tf.placeholder(tf.int32, [None, None], name="input")
        p_learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        return (p_input, p_targets, p_learning_rate)

    def get_init_cell(self, batch_size, rnn_size, layer_count=2):
        basic_lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_lstm] * layer_count)
        initial_state = tf.identity(multi_rnn_cell.zero_state(batch_size, tf.float32), name="initial_state")
        
        return (multi_rnn_cell, initial_state)
