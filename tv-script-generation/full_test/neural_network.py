import tensorflow as tf
import timeit
import datetime
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib import seq2seq

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

    def get_embed(self, input_data, vocab_size, embed_dim):
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
        return tf.nn.embedding_lookup(embedding, input_data)

    def build_rnn(self, cell, inputs):
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        final_state = tf.identity(final_state, name="final_state")
        return (outputs, final_state)


    def build_nn(self, cell, rnn_size, input_data, vocab_size):
        embed_layer = self.get_embed(input_data, vocab_size, rnn_size)
        rnn, final_state = self.build_rnn(cell, embed_layer)
        fully_connected = tf.layers.dense(rnn, units=vocab_size, activation=None)
        tf.summary.histogram('fully_connected', fully_connected)
        return (fully_connected, final_state)

    def build_model(self, int_to_vocab, rnn_size, rnn_layer_count, summary_output_dir):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            vocab_size = len(int_to_vocab)
            self.input_text, self.targets, self.lr = self.get_inputs()
            input_data_shape = tf.shape(self.input_text)
            cell, self.initial_state = self.get_init_cell(input_data_shape[0], rnn_size, layer_count=rnn_layer_count)
            logits, self.final_state = self.build_nn(cell, rnn_size, self.input_text, vocab_size)

            # Probabilities for generating words
            probs = tf.nn.softmax(logits, name='probs')

            # Loss function
            self.cost = seq2seq.sequence_loss(
                logits,
                self.targets,
                tf.ones([input_data_shape[0], input_data_shape[1]]))
            tf.summary.scalar('train_loss', self.cost)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(capped_gradients)

            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(summary_output_dir, graph=self.train_graph)

    def run_train_epoch(self, sess, batches, learning_rate, epoch_i):
        state = sess.run(self.initial_state, {self.input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                self.input_text: x,
                self.targets: y,
                self.initial_state: state,
                self.lr: learning_rate}
            train_loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], feed)

        summary = sess.run(self.merged_summaries, feed)
        self.train_writer.add_summary(summary, epoch_i)

        return train_loss

    def save_trained_model(self, sess, save_dir, epoch_number):
        saver = tf.train.Saver()
        full_save_directory = '{}/epoch_{}'.format(save_dir, epoch_number)
        if not os.path.exists(full_save_directory):
            os.makedirs(full_save_directory)
        saver.save(sess, full_save_directory)
        print('Model trained and saved to {}.'.format(full_save_directory))

    def generate_test_script(self, prime_word, train_graph, initial_state, gen_length, vocab_to_int, int_to_vocab, sess, token_dict, seq_length):
        print('Generating new text with prime word: {}'.format(prime_word))
        test_final_state, test_probs = self.get_tensors(train_graph)
        gen_sentences = [prime_word + ':']
        prev_state = sess.run(initial_state, {self.input_text: np.array([[1]])})

        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [test_probs, test_final_state],
                {self.input_text: dyn_input, initial_state: prev_state})
            
            pred_word = self.pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

            gen_sentences.append(pred_word)

        # Remove tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')
        return tv_script

    def get_tensors(self, loaded_graph):
        final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
        probabilities_tensor = loaded_graph.get_tensor_by_name("probs:0")
        return (final_state_tensor, probabilities_tensor)

    def pick_word(self, probabilities, int_to_vocab):
        to_choose_from = list(int_to_vocab.values())
        return np.random.choice(to_choose_from, p=probabilities)

    def train_model(self, batches, num_epochs, learning_rate, save_every, save_dir, test_every, prime_word, gen_length, text_processor, seq_length):
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            #print('Train graph:', train_graph.get_operations())
            print('Running {} batches per epoch.'.format(len(batches)))

            all_start_time = timeit.default_timer()
            for epoch_i in range(num_epochs):
                train_loss = self.run_train_epoch(sess, batches, learning_rate, epoch_i)

                last_end_time = timeit.default_timer()

                total_seconds_so_far = last_end_time - all_start_time
                total_time_so_far = datetime.timedelta(seconds=total_seconds_so_far)
                estimated_to_finish = datetime.timedelta(seconds=num_epochs * total_seconds_so_far / (epoch_i + 1) - total_seconds_so_far)

                print('Epoch {:>3}/{} train_loss = {:.3f}, time so far {}, estimated to finish {}'
                    .format(epoch_i + 1, num_epochs, train_loss, total_time_so_far, estimated_to_finish))

                if (epoch_i % save_every == 0 or epoch_i == num_epochs - 1):
                    self.save_trained_model(sess, save_dir, epoch_i + 1)

                if (epoch_i % test_every == 0 or epoch_i == num_epochs - 1):
                    test_final_state, test_probs = self.get_tensors(self.train_graph)
                    tv_script = self.generate_test_script(prime_word, self.train_graph, self.initial_state, gen_length, text_processor.vocab_to_int, text_processor.int_to_vocab, sess, text_processor.token_dict, seq_length)

                    print("*********************************************************************************************")
                    print(tv_script)
                    print("*********************************************************************************************")
