# word-embedder

*word-embedder* is a tool for training word embeddings, written in [Python](https://www.python.org/) and with embedding models implemented using [Tensorflow](https://www.tensorflow.org/). Currently supported embeddings models include:

* word2vec
  * Skip-gram

## Dependencies

*word-embedder* requires that the following packages are installed:

* [Tensorflow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)

## Using word-embedder

*word-embedder* is run from the command line. For example, in linux:

```bash
python3 word-embedder.py -i input_file -m word2vec -v 10000 -e 300 -ep 50000 -o output.txt
```

## Usage

### Required arguments

#### -i / --input

An input file or files. Input files are minimally processed (punctuation and symbols stripped, split on whitespace) before being used for training.

#### -m / --model

A word embedding model to use. Currently supported models include: 'word2vec'.

#### -v / --vocabsize

Vocabulary size (n-most common words from the input file(s) from which to create embeddings).

#### -e / --embedsize

Emedding size. How many dimensions should the resulting word embeddings (vectors) be.

#### -ep / --epochs

How many epochs / iterations to run the model for during training.

### Optional arguments

#### -s / --seed

Seed the graph (for deterministic training)

## Configuration for individual models

The `config.py` file in the root directory contains a number of options for each embedding model that can be altered before running *word-embedder*. For example, options for running the skipgram variant of word2vec might look like this:

```python3
config = { 
	# The word embedding algorithm to run.
	"word2vec": {
		# Model variant for running 'word2vec'
		"variant": "skipgram",

		# Choose an output function for running 'word2vec'
		"output_function": "nce",

		# Choose the number of skips and context window size
		"number_of_skips": 2,
		"context_window": 2,

		# Select a mini-batch size for training
		"batch_size": 128,

		# Choose the number of negatives to be sampled
		"negative_samples": 64
	}
}
```

## License

*word-embedder* is available under the MIT license, a copy of which is included with this repository