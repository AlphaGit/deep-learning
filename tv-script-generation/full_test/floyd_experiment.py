# -*- coding: utf-8 -*-
## Parameters

# local
'''
num_epochs = 20 # 100
batch_size = 64 #1024
rnn_size = 20 # 50
rnn_layer_count = 1 #2
seq_length = 20
learning_rate = 0.001
data_percentage = 0.01 #1
'''

# floyd
#'''
num_epochs = 100
batch_size = 1024
rnn_size = 50
rnn_layer_count = 2
seq_length = 20
learning_rate = 0.001
data_percentage = 1
#'''

save_dir = './output/model'
data_dir = 'all_lines_manual.txt'

# For example generation
test_every = 5
save_every = 10
gen_length = 200
prime_word = 'moe_szyslak'

#########################################################################

import numpy as np
import warnings
import tensorflow as tf

from text_processor import TextProcessor 
from neural_network import NeuralNetwork

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

tf.logging.set_verbosity(tf.logging.ERROR)

text_processor = TextProcessor(data_dir, data_percentage)
text = text_processor.load_and_preprocess_text()

print('Creating computation graph...')
nn = NeuralNetwork()

summary_output_dir = '/output/e{}_b{}_rnn{}_rnnl{}_seq{}_lr{}_dp{}'.format(num_epochs, batch_size, rnn_size, rnn_layer_count, seq_length, learning_rate, data_percentage)
nn.build_model(text_processor.int_to_vocab, rnn_size, rnn_layer_count, summary_output_dir)
print('Computation graph created.')

print('Training...')
batches = get_batches(text_processor.int_text, batch_size, seq_length)
nn.train_model(batches, num_epochs, learning_rate, save_every, save_dir, test_every, prime_word, gen_length, text_processor, seq_length)
