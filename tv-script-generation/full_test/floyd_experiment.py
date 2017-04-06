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
import helper
import numpy as np
import warnings
import tensorflow as tf
import timeit
import datetime

from distutils.version import LooseVersion
from collections import Counter
from tensorflow.contrib import seq2seq

#########################################################################

def create_lookup_tables(text):
    words = sorted(Counter(text), reverse=True)
    vocab_to_int = { word: idx for idx, word in enumerate(words) }
    int_to_vocab = { idx: word for word, idx in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab

#########################################################################

def token_lookup():
    return {
        ".": "||PERIOD||",
        ",": "||COMMA||",
        "\"": "||QUOTATION_MARK||",
        ";": "||SEMICOLON||",
        "!": "||EXCLAMATION_MARK||",
        "?": "||QUESTION_MARK||",
        "(": "||LEFT_PARENTHESIS||",
        ")": "||RIGHT_PARENTHESIS||",
        "--": "||DASH||",
        "\n": "||return||"
    }

#########################################################################

def get_inputs():
    p_input = tf.placeholder(tf.int32, [None, None], name="input")
    p_targets = tf.placeholder(tf.int32, [None, None], name="input")
    p_learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return (p_input, p_targets, p_learning_rate)

#########################################################################

def get_init_cell(batch_size, rnn_size, layer_count=2):
    basic_lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_lstm] * layer_count)
    initial_state = tf.identity(multi_rnn_cell.zero_state(batch_size, tf.float32), name="initial_state")
    
    return (multi_rnn_cell, initial_state)

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

def preprocess_text(text, token_dict):
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    return text

#########################################################################

def get_tensors(loaded_graph):
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    initial_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    probabilities_tensor = loaded_graph.get_tensor_by_name("probs:0")
    return (input_tensor, initial_state_tensor, final_state_tensor, probabilities_tensor)

#########################################################################

tf.logging.set_verbosity(tf.logging.ERROR)

text = helper.load_data(data_dir)

text_lines_to_use = int(len(text) * data_percentage)
print("Data: Using {:,} out of {:,} total lines available".format(text_lines_to_use, len(text)))
text = text[:text_lines_to_use]

token_dict = token_lookup()
text = preprocess_text(text, token_dict)
vocab_to_int, int_to_vocab = create_lookup_tables(text)
int_text = [vocab_to_int[word] for word in text]

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

print('Creating computation graph...')
train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, layer_count=rnn_layer_count)
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

    batches = get_batches(int_text, batch_size, seq_length)

print('Computation graph created.')

print('Training...')

all_start_time = timeit.default_timer()

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    #print('Train graph:', train_graph.get_operations())
    print('Running {} batches per epoch.'.format(len(batches)))

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        last_start_time = timeit.default_timer()

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

        last_end_time = timeit.default_timer()

        total_seconds_so_far = last_end_time - all_start_time
        total_time_so_far = datetime.timedelta(seconds=total_seconds_so_far)
        estimated_to_finish = datetime.timedelta(seconds=num_epochs * total_seconds_so_far / (epoch_i + 1) - total_seconds_so_far)

        print('Epoch {:>3}/{} train_loss = {:.3f}, time so far {}, estimated to finish {}'
            .format(epoch_i + 1, num_epochs, train_loss, total_time_so_far, estimated_to_finish))

        summary = sess.run(merged_summaries, feed)
        train_writer.add_summary(summary, epoch_i)

        if (epoch_i % save_every == 0 or epoch_i == num_epochs - 1):
            # Save Model
            saver = tf.train.Saver()
            full_save_directory = '{}/epoch_{}'.format(save_dir, epoch_i)
            if not os.path.exists(full_save_directory):
                os.makedirs(full_save_directory)
            saver.save(sess, full_save_directory)
            last_save_directory = full_save_directory
            print('Model trained and saved to {}.'.format(full_save_directory))

        if (epoch_i % test_every == 0 or epoch_i == num_epochs - 1):
            print('Generating new text with prime word: {}'.format(prime_word))
            _, _, test_final_state, test_probs = get_tensors(train_graph)
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
            
            print("*********************************************************************************************")    
            print(tv_script)
            print("*********************************************************************************************")    
