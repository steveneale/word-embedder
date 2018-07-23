#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'config.py'

Configuration for running 'word_embedder.py'.

2017 Steve Neale <steveneale3000@gmail.com>

"""

config = { # The word embedding algorithm to run. This will have been decided at the command line when running 'word_embedder.py'
		   "word2vec": {
		   				 # Model variant for running 'word2vec'. The available variants are:
		   				 # --- 'skipgram'
		   				 "variant": "skipgram",

						 # Choose an output function for running 'word2vec'. Supported functions include:
						 # --- 'nce' (noise contrastive estimation)
						 # --- 'softmax'
						 "output_function": "nce",

						 # Choose the number of skips and context window size when using the skip-gram method
						 "number_of_skips": 2,
						 "context_window": 2,

						 # Select a mini-batch size for training
						 "batch_size": 128,

						 # Choose the number of negatives to be sampled (during noise contrastive estimation)
						 "negative_samples": 64
			 		   }
		 }
