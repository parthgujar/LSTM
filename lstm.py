import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time



def time_elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dataset(words):

    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def RNN(x, weights, biases,n_input,n_hidden):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicRNNCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


start_time = time.time()

logs_path = '/home/kdeshmukh/Dropbox/Fall_17/256_polett/HW/5_1/rnn_logs'
writer = tf.summary.FileWriter(logs_path)

training_file = '/home/kdeshmukh/Dropbox/Fall_17/256_polett/HW/5_1/input'

training_data = read_data(training_file)
print training_data
dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = input("Enter learning rate : ") # 0.001
training_iters = input("Enter training inerations :  ")# 15000
display_step = 1000
n_input = 2
# number of units in RNN cell
n_hidden = 100

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])
# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

pred = RNN(x, weights, biases,n_input,n_hidden)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
    # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)

        symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],
                                                    feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Loss= " +
                  "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " +
                  "{:.2f}%".format(100 * acc_total / display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
        step += 1
        offset += (n_input + 1)
    print("Optimization Finished!")
    print("Elapsed time: ", time_elapsed(time.time() - start_time))


    predicted_out_text = []

    predicted_matrix = []

    for i in range(len(training_data)):

        symbols_in_keys = dictionary[str(training_data[i])]

        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        prediction = session.run(pred,feed_dict={x: symbols_in_keys})

        predicted_matrix.append(prediction)

        pred_word = reverse_dictionary[int(tf.argmax(prediction,1).eval())]

        predicted_out_text.append(pred_word)


print predicted_matrix,training_data,reverse_dictionary.values()
