import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import random
import utils
import time
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
display_step = 1000
num_hidden = 512
num_input = 3
num_layers = 1
max_grad_norm = 5


def build_lstm(x, num_input, words_size):
    weights = {
        'weight': tf.Variable(tf.random_normal([num_hidden, words_size]))
    }
    biases = {
        'bias': tf.Variable(tf.random_normal([words_size]))
    }
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, num_input])

    # Generate a num_input-element sequence of inputs
    x = tf.split(x, num_input, 1)

    rnn_cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(num_layers)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x,
                                     dtype=tf.float32)

    # there are num_input outputs but
    # we only want the last outputtf.contrib.rnn.static_bidirectional_rnn
    return tf.matmul(outputs[-1], weights['weight']) + biases['bias']


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)


def get_weights():
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
            v.name.endswith('weights:0')]


def graph(x, y1, y2):
    plt.xlabel("Max Updates")
    plt.ylabel("Cost-Accuracy")
    acc_line, = plt.plot(x, y1, color='blue',
                         label="Accuracy",
                         linestyle='dashed')
    training_line, = plt.plot(x, y2, color='green',
                              label="Training cost ", linestyle='dashed')
    plt.legend(handles=[acc_line, training_line], loc=2)
    plt.show()


def plot_test_accuracy(counts):
    plt.xlabel("No. of words")
    plt.ylabel("Accuracy")
    accuracy = []
    for i in range(len(counts)):
        accuracy.append(counts[:(i + 1)].count(1) * 1.0 / (i + 1))
    plt.plot(range(1, len(counts) + 1), accuracy)
    plt.show()


def train(train_data_file, model_file, max_update, regularization='L1',
          learning_rate=0.001):
    training_data = utils.read_data(train_data_file)
    # print training_data
    encode, decode = utils.build_encode_decode_dictionary(training_data)
    words_size = len(encode)

    x = tf.placeholder("float", [None, num_input, 1])
    y = tf.placeholder("float", [None, words_size])

    pred = build_lstm(x, num_input, words_size)
    start_time = time.time()

    # Loss and optimizer
    if regularization == 'L1':
        l1_regularizer = tf.contrib.layers.l1_regularizer(
            scale=0.005, scope=None
        )
        weights = tf.trainable_variables()  # all vars of your graph
        regularization_cost = tf.contrib.layers.apply_regularization(
            l1_regularizer, weights)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=y)) + regularization_cost
    elif regularization == 'L2':
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=0.005, scope=None)
        weights = tf.trainable_variables()  # all vars of your graph
        regularization_cost = tf.contrib.layers.apply_regularization(
            l2_regularizer, weights)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=y)) + regularization_cost
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate). \
        minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate). \
    #     minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    step_history = []
    cost_history = []
    acc_history = []
    with tf.Session() as session:
        writer = tf.summary.FileWriter('./graphs', session.graph)
        session.run(init)
        step = 0
        offset = random.randint(0, num_input + 1)
        end_offset = num_input + 1
        acc_total = 0
        loss_total = 0

        while step < max_update:
            # Generate a mini-batch. Add some randomness on selection process.
            if offset > (len(training_data) - end_offset):
                offset = random.randint(0, num_input + 1)

            words_freq = [[encode[str(training_data[i])]] for i in
                          range(offset, offset + num_input)]
            words_freq = np.reshape(np.array(words_freq),
                                    [-1, num_input, 1])

            onehot_output = np.zeros([words_size], dtype=float)
            onehot_output[
                encode[str(training_data[offset + num_input])]] = 1.0
            onehot_output = np.reshape(onehot_output, [1, -1])

            _, acc, loss, onehot_pred = session.run(
                [optimizer, accuracy, cost, pred], feed_dict=
                {x: words_freq, y: onehot_output})
            loss_total += loss
            acc_total += acc
            if (step + 1) % display_step == 0:
                print "Step= ", str(step + 1), ", Avg loss= ", \
                    (loss_total / display_step), ", Avg accuracy= ", \
                    (acc_total / display_step)
                step_history.append(step)
                cost_history.append(loss_total / display_step)
                acc_history.append(acc_total / display_step)
                acc_total = 0
                loss_total = 0
                plays_in = [training_data[i] for i in range(
                    offset, offset + num_input)]
                plays_out = training_data[offset + num_input]
                play_out_pred = decode[int(tf.argmax(onehot_pred, 1).eval())]
                print "%s - [%s] vs [%s]" % (
                    plays_in, plays_out, play_out_pred)
            step += 1
            offset += (num_input + 1)
        print "Training completed!"
        print "Training time: ", time.time() - start_time
        saver = tf.train.Saver()
        save_path = saver.save(session, os.path.join(os.getcwd(), model_file))
        # saver.save(session, "model1")

        graph(step_history, acc_history, cost_history)


def test(data_file, model_file, sample_text, newtext_length):
    training_data = utils.read_data(data_file)
    # print training_data
    encode, decode = utils.build_encode_decode_dictionary(training_data)
    words_size = len(encode)

    x = tf.placeholder("float", [None, num_input, 1])
    y = tf.placeholder("float", [None, words_size])

    pred = build_lstm(x, num_input, words_size)

    start_time = time.time()
    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver()
        saver.restore(session, model_file)

        sample_text = sample_text.strip()
        words = sample_text.split(' ')
        try:
            symbols_in_keys = [encode[str(words[i])] for i in
                               range(len(words))]
            for i in range(newtext_length):
                keys = np.reshape(np.array(symbols_in_keys),
                                  [-1, num_input, 1])
                onehot_pred = session.run(pred,
                                          feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sample_text = "%s %s" % (
                    sample_text, decode[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sample_text)
            predictions = sample_text.split(" ")
            original_text = predictions[:3]
            predictions = predictions[3:]
            index = utils.find_index(training_data, original_text)
            if index == -1:
                print "No such original text find. Hence accuracy cannot be " \
                      "calculated."
            else:
                count_correct = []
                for i in range(len(predictions)):
                    if predictions[i] == training_data[index + i]:
                        count_correct.append(1)
                    else:
                        count_correct.append(0)
                print "Test Accuracy :", (count_correct.count(
                    1) * 1.0) / len(count_correct)
                plot_test_accuracy(count_correct)
        except:
            print("Word not in the encoded dictionary")

    print 'Testing completed!'
    print 'Testing time:', time.time() - start_time


if __name__ == "__main__":
    mode = ''
    max_update = ''
    data_file = ''
    learning_rate = ''
    model_file = ''
    args = sys.argv
    mode = args[1]
    if mode == 'train':
        if len(args) != 7:
            print ('6 Arguments expected for training mode: <mode> <data_file>'
                   ' <model_file> '
                   '<max_update><regularization><learning_rate>')
            exit(0)
        data_file = args[2]
        model_file = args[3]
        max_update = int(args[4])
        regularization = args[5]
        learning_rate = float(args[6])
        if regularization == 'l1' or regularization == 'L1':
            regularization = 'L1'
        elif regularization == 'l2' or regularization == 'L2':
            regularization = 'L2'
        elif regularization.lower() == 'none':
            regularization = 'none'
        else:
            sys.exit("Invalid value in regularization! allowed values are: "
                     "l1,l2,none")
        train(data_file, model_file, max_update, regularization, learning_rate)
    elif mode == 'test':
        if len(args) != 6:
            print ('5 Arguments expected for testing mode: <mode> '
                   '<data_file> <model_file> <sample_text> '
                   '<newtext_length>')
            exit(0)
        data_file = args[2]
        model_file = args[3]
        sample_text = args[4]
        newtext_length = int(args[5])
        test(data_file, model_file, sample_text, newtext_length)
    else:
        print 'Invalid argument.'
