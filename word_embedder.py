#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'word_embedder.py'

Train a set of word embeddings (vector space representation of words)

2018 Steve Neale <steveneale3000@gmail.com>

"""

import sys
import os
import re
import math
import random
import collections

import argparse

import numpy as np
import tensorflow as tf

from config import config


""" Global Variables """

embedding_size = 0
vocabulary_size = 0

input_corpus = []
corpus_data = []
vocabulary = {}

output_filename = ""


""" Seeds (for deterministic traning) """

SEED = 1985
random.seed(SEED)
np.random.seed(SEED)


""" Primary Functions """

def load_input_words(input_text):
    """ Convert the given input text(s) to a vocabulary """

    input_words = []
    # Loop through the input text(s) and open each of them
    for text in input_text:
        with open(text) as loaded_input:
            # Split the input text into lines and loop through them
            lines = loaded_input.read().splitlines()
            for line in lines:
                # Remove punctuation from each line, split it into words, and add each word to the vocabulary
                line = re.sub(r"[^\w\s]", "", line, re.UNICODE)
                words = line.split()
                for word in words:
                    input_words.append(word.lower())
    return input_words


def build_dataset(input_words, vocabulary_size):
    """ Build the appropriate datasets for the desired vocabulary size given a corpus of input words """

    # Extract the n-most common words from the input (given a desired vocabulary size)
    vocabulary_words = collections.Counter(input_words).most_common(vocabulary_size)
    # Create a dictionary of indexes for words in the vocabulary, in descending order of most common
    vocab = dict()
    for word, _ in vocabulary_words:
        vocab[word] = len(vocab) + 1
    # Create the input corpus data - the index in the vocabulary of every known word in the input text
    corpus_data = list()
    for word in input_words:
        if word in vocab:
            index = vocab[word]
        else:
            index = 0
        corpus_data.append(index)
    return corpus_data, dict(zip(vocab.values(), vocab.keys()))


def skipgram_batches(corpus_data, batch_size, num_skips, context_window):
    """ Generate training data mini-batches using the skip-gram method """

    buffer_index = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * context_window
    # Define n-dimensional arrays to hold the input words and the words from the context window
    input_words = np.ndarray(shape=(batch_size), dtype=np.int32)
    context_words = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # Define the span - the length of the context window either side of the input word (1)
    span = 2 * context_window + 1
    # Create a span-sized buffer that appends the word at the current index until the length of the corpus data is reached
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(corpus_data[buffer_index])
        buffer_index = (buffer_index + 1) % len(corpus_data)
    for i in range(batch_size // num_skips):
        # Define the initial 'target' (input word at the centre of the span) and add it to the 'targets to avoid' 
        target = context_window
        targets_to_avoid = [context_window]
        for j in range(num_skips):
            # Create a new 'target' (context) word at random and if it's not already in the list of targets to avoid, add it
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            # Add the input from the centre of the context window and the currently targeted context word to their respective arrays
            input_words[i * num_skips + j] = buffer[context_window]
            context_words[i * num_skips + j, 0] = buffer[target]
        # Add the current word to the buffer and increment the buffer index
        buffer.append(corpus_data[buffer_index])
        buffer_index = (buffer_index + 1) % len(corpus_data)
    buffer_index = (buffer_index + len(corpus_data) - span) % len(corpus_data)
    return input_words, context_words


def run_embedding_model(model, epochs, seed=False):
    """ Build and run a given word embeddings model """

    if model == "word2vec":
        embedding_model = tf.Graph()
        with embedding_model.as_default():
            # Seed the embedding model with values (for deterministic training), if required
            if seed == True:
                tf.set_random_seed(SEED)
            # Create variables to hold the inputs and labels for training
            training_inputs = tf.placeholder(tf.int32, shape=(config["word2vec"]["batch_size"]))
            training_labels = tf.placeholder(tf.int32, shape=(config["word2vec"]["batch_size"], 1))     
            # Define the shape of the embeddings, and lookup the embeddings layer for each training input
            embedding_shape = tf.Variable(tf.random_uniform([len(vocabulary), embedding_size], -1.0, 1.0))
            embeddings = tf.nn.embedding_lookup(embedding_shape, training_inputs)
            # Define weights (random values from a truncated normal distribution) and biases (zeros) for the model, 
            weights = tf.Variable(tf.truncated_normal([embedding_size, vocabulary_size], stddev=1.0 / math.sqrt(embedding_size)))
            biases = tf.Variable(tf.zeros([vocabulary_size]))
            # Multiply the weights and the embeddings, and add the biases to form a (hidden) output layer
            hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embeddings))) + biases
            # Produce a one hot vector of output labels, and compute the average softmax cross entropy between the (hidden) output layer and the one hot labels
            one_hot_labels = tf.one_hot(training_labels, vocabulary_size)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hidden_out, labels=one_hot_labels))
            # Define a stochastic gradient descent optimiser, depending on the output function type (softmax of NCE)
            if config["word2vec"]["output_function"] == "softmax":
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
            elif config["word2vec"]["output_function"] == "nce":
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
                nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                         biases=nce_biases,
                                                         labels=training_labels,
                                                         inputs=embeddings,
                                                         num_sampled=config["word2vec"]["negative_samples"],
                                                         num_classes=vocabulary_size))
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)
            # Normalise the embeddings
            normalized_embeddings = embedding_shape / tf.sqrt(tf.reduce_sum(tf.square(embedding_shape), 1, keepdims=True))
            # Initialise the global variables
            initializer = tf.global_variables_initializer()
            # If seeding and deterministic training are required, set the session configuration to use only one thread for training
            session_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) if seed == True else None
            # Start and initialise a tensorflow session using the embedding model
            with tf.Session(graph=embedding_model, config=session_config) as session:
                initializer.run()
                average_loss = 0
                # For each epoch...
                for step in range(epochs):
                    # Generate batches of training data and labels and use them to construct a feed dictionary for the model 
                    batch_inputs, batch_labels = skipgram_batches(corpus_data, 
                                                                  config["word2vec"]["batch_size"], 
                                                                  config["word2vec"]["number_of_skips"], 
                                                                  config["word2vec"]["context_window"])
                    feed_dictionary = {training_inputs: batch_inputs, training_labels: batch_labels}
                    # Run the model using the feed dictionary, and add the returned loss value to the average loss
                    _, loss_value = session.run([optimizer, cross_entropy], feed_dict=feed_dictionary)
                    average_loss += loss_value
                    # Every 2000 epochs, compute and print the average loss
                    if step % 2000 == 0:
                        if step > 0:
                            average_loss /= 2000
                        print("Average loss at step {}: {}".format(step, average_loss))
                        average_loss = 0
                # Compute the value of the (normalised) embeddings
                final_embeddings = normalized_embeddings.eval()
                # Print the final embeddings to an output file
                output_file = open("{}/output/{}.txt".format(os.path.dirname(os.path.abspath(__file__)), output_filename), "w")
                print("{} {}".format(vocabulary_size, embedding_size), file=output_file)
                for i in range(vocabulary_size):
                    vectors = " ".join([str(vector) for vector in final_embeddings[i]])
                    print("{} {}".format(vocabulary[i+1], vectors), file=output_file)
                return final_embeddings


def parse_arguments(arguments):
    """ Parse command line arguments """

    # Create an argument parser, and pop the optional arguments section
    parser = argparse.ArgumentParser(description="word_embedder - train sets of word embeddings (vector space representations of words)")
    optional = parser._action_groups.pop()  
    # Add a new section with the required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("-i", "--input", help="Input file path(s)", nargs="+", required=True)
    required.add_argument("-m", "--model", help="Model ('word2vec')", required=True)
    required.add_argument("-v", "--vocabsize", help="Vocabulary size", required=True)
    required.add_argument("-e", "--embedsize", help="Embedding size", required=True)
    required.add_argument("-ep", "--epochs", help="Number of epochs", required=True)
    required.add_argument("-o", "--output", help="Output filename", required=True)
    # Re-attach the optional arguments section (after the required arguments)
    optional.add_argument("-s", "--seed", help="Seed the graph (for deterministic behaviour)", action="store_true")
    parser._action_groups.append(optional)
    return(parser.parse_args())


""" Main Function (called when 'word_embedder.py' is run from the command line) """

if __name__ == "__main__":
    """ Run the program, depending on arguments given """

    # Process the given command line arguments and populate the necessary global variables
    arguments = parse_arguments(sys.argv[1:])
    embedding_size, vocabulary_size, output_filename = int(arguments.embedsize), int(arguments.vocabsize), arguments.output
    # Load a corpus of input words from the input text(s) 
    input_corpus = load_input_words(arguments.input)
    # Produce datasets from the input corpus
    corpus_data, vocabulary = build_dataset(input_corpus, vocabulary_size)
    # Run the chosen word embedding model
    embeddings = run_embedding_model(arguments.model, int(arguments.epochs), arguments.seed)
