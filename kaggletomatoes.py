import pandas as pd
from collections import Counter
import random
import math

PAD_INDEX = 0
UNKNOWN_INDEX = 1
# http://neuralnetworksanddeeplearning.com/chap3.html
train = pd.read_csv("data/train.tsv", header=0, delimiter="\t", quoting=3)

num_phrases = train["PhraseId"].size

training_sentiment = []

# Fast Fola suggested this naming convention
def hot_vectorize(sentiment):
    one_hot_vector = [0,0,0,0,0]
    one_hot_vector[sentiment-1]=1
    return one_hot_vector

sentences = []
last_sentence_id = 0
for i in range(0, num_phrases):
    sentence_id = train["SentenceId"][i]
    if sentence_id != last_sentence_id:
        sentences.append(train["Phrase"][i].split())
        last_sentence_id = sentence_id
        training_sentiment.append(hot_vectorize(int(train["Sentiment"][i])))


print(sentences[0:1])

print("Hot vectorized Sentiment is: ", training_sentiment[0:2])

sentence_max = 0
counter = Counter()
for sentence in sentences:
    sentence_max = max(sentence_max, len(sentence))
    for word in sentence:
        counter[word] += 1

print("Sentence max :" + str(sentence_max))
print(len(counter))
print(counter.most_common(10))

i = 2
lookup_table = {}
for word, _ in counter.most_common(18000):
    lookup_table[word] = i
    i += 1

print(lookup_table["the"])

def lookup_word(word):
    if word in lookup_table:
        return lookup_table[word]
    else:
        return UNKNOWN_INDEX
    # return lookup_table[word] if word in lookup_table else UNKNOWN_INDEX

sentence_input = []
for sentence in sentences:
    numeric_words = list(map(lookup_word, sentence))
    numeric_words += [PAD_INDEX] * (sentence_max - len(numeric_words))
    sentence_input.append(numeric_words)

# print(sentence_input[0:2])

# Build the neural network itself.

def generate_layer(input_size, output_size):
    weights = []
    biases = []
    for i in range(0,output_size):
        weights_row = []
        biases.append(random.random() * 2 - 1)
        for j in range(0, input_size):
            weights_row.append(random.random() * 2 - 1)
        weights.append(weights_row)

    return weights, biases

def evaluate_layer(weights, biases, inputs, apply_function):
    layer = []
    for i in range(0, len(biases)):
        weightedSum = 0
        for j in range(0, len(inputs)):
            weightedSum += (weights[i][j]*inputs[j])
        weightedSum += biases[i]
        layer.append(weightedSum)
    return apply_function(layer)

def activation_function(layer):
    return map(math.tanh,layer)

def transfer_function(layer):
    numerator = map(math.exp, layer)
    denominator = sum(numerator)
    return map(lambda x: x/denominator, numerator)


def cross_entropy(expected, actual):
    error_vector = []
    for i in range(0, len(expected)):
        error_vector.append(expected[i] * math.log(actual[i]))
    return -sum(error_vector)

def derivative_cross_entropy(expected, actual):
    derivative = []
    for i in range(0, len(expected)):
        derivative.append(actual[i] - expected[i])
    return derivative

def derivative_softmax(layer):
    # FIXME: Finish this!
    return []

hidden_layer_size = 800
hidden_weights, hidden_biases = generate_layer(sentence_max, hidden_layer_size)

# print(hidden_weights[0:2])

# Naming our first hidden layer nodes h1
# Note to future team : Fast Eric made us do this
h1 = evaluate_layer(hidden_weights, hidden_biases, sentence_input[0],activation_function)

# print(h1[0:2])

output_layer_size = 5
output_weights, output_biases = generate_layer(hidden_layer_size, output_layer_size)

# print("The output weights are ",  output_weights[0:2])

y = evaluate_layer(output_weights, output_biases, h1, transfer_function)

print("Output layer is ", y)

print("Sum of y is: ", sum(y))

print("cross_entropy result", cross_entropy(training_sentiment[0], y))

print("cross_entropy fake result", cross_entropy(y, y))

# TODO: Add training based on cost function.  Possibly add another hidden layer (h2!), then word2vec.
