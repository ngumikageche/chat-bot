import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
"""Loading our json data"""
import json
with open('intents.json') as file:
    data = json.load(file)

"""Data extraction"""
words = []
labels = []
docs_x = []
docs_y = []

"""Loop through our JSON data and extract the data we want.
 For each pattern we will turn it into a list of words using nltk.word_tokenizer,
   rather than having them as strings.
   """

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent['tag'])

"""Word Stemming
attempting to find the root of the word
example word "thats" stem might be "that"

This code will simply create a unique list
 of stemmed words to use in the next step of
   our data preprocessing."""

words  = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

"""Bag of words
"""

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in words:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

"""Develop a model"""
tensorflow.reset_dafault_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0], activation="softmax"))
net = tflearn.regression(net)

model = tflearn.DNN(net)

"""Training our Model"""

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
