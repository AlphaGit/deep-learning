# -*- coding: utf-8 -*-
## Parameters

num_epochs = 20 # 100
batch_size = 64 #1024
rnn_size = 20 # 50
rnn_layer_count = 1 #2
seq_length = 20
learning_rate = 0.001
data_percentage = 0.01 #1

save_dir = './output/model'
data_dir = 'all_lines_manual.txt'

# For example generation
test_every = 5
save_every = 10
gen_length = 200
prime_word = 'moe_szyslak'

#########################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import warnings
import tensorflow as tf
import timeit
import datetime

from text_processor import TextProcessor 
from neural_network import NeuralNetwork

from distutils.version import LooseVersion
from tensorflow.contrib import seq2seq

#########################################################################

def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    return tf.nn.embedding_lookup(embedding, input_data)

#########################################################################

def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")
    return (outputs, final_state)

#########################################################################

def build_nn(cell, rnn_size, input_data, vocab_size):
    embed_layer = get_embed(input_data, vocab_size, rnn_size)
    rnn, final_state = build_rnn(cell, embed_layer)
    fully_connected = tf.layers.dense(rnn, units=vocab_size, activation=None)
    tf.summary.histogram('fully_connected', fully_connected)
    return (fully_connected, final_state)

#########################################################################

def get_batches(int_text, batch_size, seq_length):
    total_sequences = len(int_text) // seq_length
    
    fixed_ints = int_text[:seq_length * total_sequences]
    
    result = []
    current_batch_input = []
    current_batch_output = []
    read_sequences_count = 0
    for index in range(0, len(fixed_ints), seq_length):
        batch_input = fixed_ints[index : index + seq_length] # take [x, x+1, x+2, ..., x+seq_length-1] -> seq_length elements
        batch_output = fixed_ints[index + 1 : index + seq_length + 1] # take [x+1, x+2, ..., x+seq_length] -> seq_length elements
        
        current_batch_input.append(batch_input)
        current_batch_output.append(batch_output)

        read_sequences_count += 1
        # It is possible we don't complete a batch. In that case, this if won't execute and the result won't be added.
        if read_sequences_count == batch_size:
            result.append([ current_batch_input, current_batch_output ])
            current_batch_input = []
            current_batch_output = []
            read_sequences_count = 0
    
    return np.array(result)

#########################################################################

def pick_word(probabilities, int_to_vocab):
    to_choose_from = list(int_to_vocab.values())
    return np.random.choice(to_choose_from, p=probabilities)

#########################################################################

def get_tensors(loaded_graph):
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    probabilities_tensor = loaded_graph.get_tensor_by_name("probs:0")
    return (final_state_tensor, probabilities_tensor)

#########################################################################

def generate_test_script(prime_word, initial_state, gen_length, vocab_to_int, int_to_vocab, sess, token_dict):
    print('Generating new text with prime word: {}'.format(prime_word))
    test_final_state, test_probs = get_tensors(train_graph)
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [test_probs, test_final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
    return tv_script
    
#########################################################################

def save_trained_model(save_dir, epoch_number):
    saver = tf.train.Saver()
    full_save_directory = '{}/epoch_{}'.format(save_dir, epoch_number)
    if not os.path.exists(full_save_directory):
        os.makedirs(full_save_directory)
    saver.save(sess, full_save_directory)
    print('Model trained and saved to {}.'.format(full_save_directory))

#########################################################################

def run_train_epoch(sess, initial_state, batches, learning_rate, cost, final_state, train_op, epoch_i, merged_summaries, train_writer):
    state = sess.run(initial_state, {input_text: batches[0][0]})

    for batch_i, (x, y) in enumerate(batches):
        feed = {
            input_text: x,
            targets: y,
            initial_state: state,
            lr: learning_rate}
        train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

    summary = sess.run(merged_summaries, feed)
    train_writer.add_summary(summary, epoch_i)

    return train_loss

#########################################################################

tf.logging.set_verbosity(tf.logging.ERROR)

text_processor = TextProcessor(data_dir, data_percentage)
text = text_processor.load_and_preprocess_text()

print('Creating computation graph...')
train_graph = tf.Graph()
nn = NeuralNetwork()
with train_graph.as_default():
    vocab_size = len(text_processor.int_to_vocab)
    input_text, targets, lr = nn.get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = nn.get_init_cell(input_data_shape[0], rnn_size, layer_count=rnn_layer_count)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))
    tf.summary.scalar('train_loss', cost)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)

    merged_summaries = tf.summary.merge_all()
    writer_dirname = '/output/e{}_b{}_rnn{}_rnnl{}_seq{}_lr{}_dp{}'.format(num_epochs, batch_size, rnn_size, rnn_layer_count, seq_length, learning_rate, data_percentage)
    train_writer = tf.summary.FileWriter(writer_dirname, graph=train_graph)

    batches = get_batches(text_processor.int_text, batch_size, seq_length)

print('Computation graph created.')

print('Training...')

all_start_time = timeit.default_timer()

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    #print('Train graph:', train_graph.get_operations())
    print('Running {} batches per epoch.'.format(len(batches)))

    for epoch_i in range(num_epochs):
        train_loss = run_train_epoch(sess, initial_state, batches, learning_rate, cost, final_state, train_op, epoch_i, merged_summaries, train_writer)

        last_end_time = timeit.default_timer()

        total_seconds_so_far = last_end_time - all_start_time
        total_time_so_far = datetime.timedelta(seconds=total_seconds_so_far)
        estimated_to_finish = datetime.timedelta(seconds=num_epochs * total_seconds_so_far / (epoch_i + 1) - total_seconds_so_far)

        print('Epoch {:>3}/{} train_loss = {:.3f}, time so far {}, estimated to finish {}'
            .format(epoch_i + 1, num_epochs, train_loss, total_time_so_far, estimated_to_finish))

        if (epoch_i % save_every == 0 or epoch_i == num_epochs - 1):
            save_trained_model(save_dir, epoch_i + 1)

        if (epoch_i % test_every == 0 or epoch_i == num_epochs - 1):
            test_final_state, test_probs = get_tensors(train_graph)
            tv_script = generate_test_script(prime_word, initial_state, gen_length, text_processor.vocab_to_int, text_processor.int_to_vocab, sess, text_processor.token_dict)

            print("*********************************************************************************************")
            print(tv_script)
            print("*********************************************************************************************")
