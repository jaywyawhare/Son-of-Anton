import random
import tensorflow as tf
import tflearn
from nltk.stem.lancaster import LancasterStemmer
import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

stemmer = LancasterStemmer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

with open('dataset.txt', 'r') as f:
    dataset = f.read()

words = nltk.word_tokenize(dataset)
words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))

training = []
output = []
out_empty = [0 for _ in range(len(classes))]
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = out_empty[:]
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

model.load('model.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = tokenizer.tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))
