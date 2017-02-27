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

def activation_derivative(layer):
    return map(lambda x: 1 - (math.tanh(x)**2), layer)

def transfer_function(layer):
    numerator = map(math.exp, layer)
    denominator = sum(numerator)
    return map(lambda x: x/denominator, numerator)


def cross_entropy(expected, actual):
    error_vector = []
    for i in range(0, len(expected)):
        error_vector.append(expected[i] * math.log(actual[i]))
    return -sum(error_vector)

def derivative_cross_entropy_with_softmax(expected, actual):
    derivative = []
    for i in range(0, len(expected)):
        derivative.append(actual[i] - expected[i])
    return derivative


class Layer:
    # this changed to take an input batch
    def __init__(self, input_batch, weights, biases, activation_derivative):
        self.input_batch = input_batch
        self.weights = weights
        self.biases = biases
        self.activation_derivative = activation_derivative

# Returns an array of (bias_derivatives, weight_derivatives) tuples, one per layer in reverse order.
def bias_weight_layer_derivatives(expected_outputs, actual_outputs, sentence_index, layers):
    derivatives_per_layer = []
    error = layers[0].activation_derivative(expected_outputs, actual_outputs)
    bias_derivatives = error
    weight_derivatives = []
    for neuron_error in error:
        weight_derivative_row = []
        for input in layers[0].input_batch[sentence_index]:
            weight_derivative_row.append(input * neuron_error)
        weight_derivatives.append(weight_derivative_row)
    derivatives_per_layer.append((bias_derivatives, weight_derivatives))

    # I need a Numpy
    l = 1
    while l < len(layers):
        previous_error = error  # really the next layer in a feed forward sense
        previous_layer = layers[l - 1]
        layer = layers[l]
        error = []
        for i in range(0, len(layer.weights)):
            neuron_error = 0
            for j in range(0, len(previous_layer.weights)):
                neuron_error += previous_layer.weights[j][i] * previous_error[j]
            error.append(neuron_error)

        # Compute Wx + b (z in the neural networks book)
        wx_b = []
        for i in range(0, len(layer.biases)):
            weightedSum = 0
            for j in range(0, len(layer.input_batch[sentence_index])):
                weightedSum += (layer.weights[i][j]*layer.input_batch[sentence_index][j])
            weightedSum += layer.biases[i]
            wx_b.append(weightedSum)

        derivative = layer.activation_derivative(wx_b)
        for i in range(0, len(error)):
            error[i] *= derivative[i]

        # Don't judge, we'll clean this up later.  This is totally copied and pasted from the above.

        bias_derivatives = error
        weight_derivatives = []
        for neuron_error in error:
            weight_derivative_row = []
            for input in layer.input_batch[sentence_index]:
                weight_derivative_row.append(input * neuron_error)
            weight_derivatives.append(weight_derivative_row)

        derivatives_per_layer.append((bias_derivatives, weight_derivatives))

        l += 1

    return derivatives_per_layer


# Layers is a tuple of (inputs, weights, biases) for each layer.
# layers[0] is the output layer, working backwards from there.
def backprop(expected_output_batch, actual_output_batch, layers, learning_rate = 0.001):
    # Compute partial derivatives for the biases and weights on the output layer.
    # TODO: figure out how to iterate and what args to pass to the function below
    total_bias_derivatives = []
    total_weight_derivatives = []
    for sentence_index in range(0, len(expected_output_batch)):
        bwd = bias_weight_layer_derivatives(expected_output_batch[sentence_index], actual_output_batch[sentence_index], sentence_index, layers)
        # sum this in an accumulator
        for layer_index, layer in enumerate(bwd):
            bias_derivatives, weight_derivatives = layer
            #if (sentence_index = 0):
            #    total_bias_derivatives.append(bias_derivatives)
            if sentence_index == 0:
                total_bias_derivatives.append(bias_derivatives)
                total_weight_derivatives.append(weight_derivatives)
            else:
                total_layer_bias_derivatives = total_bias_derivatives[layer_index]
                for i in range(0, len(bias_derivatives)):
                    total_layer_bias_derivatives[i] += bias_derivatives[i]

                total_layer_weight_derivatives = total_weight_derivatives[layer_index]
                for i in range(0, len(weight_derivatives)):
                    for j in range(0, len(weight_derivatives[i])):
                        total_layer_weight_derivatives[i][j] += weight_derivatives[i][j]

    batch_size = len(expected_output_batch)

    for total_layer_bias_derivatives in total_bias_derivatives:
        for i in range(0, len(total_layer_bias_derivatives)):
            total_layer_bias_derivatives[i] /= batch_size
    average_bias_derivatives = total_bias_derivatives

    for total_layer_weight_derivatives in total_weight_derivatives:
        for i in range(0, len(total_layer_weight_derivatives)):
            for j in range(0, len(total_layer_weight_derivatives[i])):
                total_layer_weight_derivatives[i][j] /= batch_size
    average_weight_derivatives = total_weight_derivatives

    # Apply derivatives to biases and weights in each layer, multiplied by learning rate
    # The learning rate is the fraction by which we are moving down the gradient of the cost function.

    for layer_index, layer in enumerate(layers):
        bias_derivatives = average_bias_derivatives[layer_index]
        weight_derivatives = average_weight_derivatives[layer_index]

        for i in range(0, len(layer.biases)):
            layer.biases[i] -= learning_rate * bias_derivatives[i]

        for i in range(0, len(layer.weights)):
            weight_row = layer.weights[i]
            for j in range(0, len(weight_row)):
                layer.weights[i][j] -= learning_rate * weight_derivatives[i][j]


# Define the network.

# hidden_layer_size = 800
hidden_layer_size = 100
hidden_weights, hidden_biases = generate_layer(sentence_max, hidden_layer_size)

# print(hidden_weights[0:2])

output_layer_size = 5
output_weights, output_biases = generate_layer(hidden_layer_size, output_layer_size)

def train_all_sentences(batch_size = 50, num_epochs = 3):
    training_data = zip(sentence_input, training_sentiment)
    for epoch_num in range(0, num_epochs):
        random.shuffle(training_data)
        for batch_num in range(0, int(math.ceil(len(training_data) / batch_size))):
            batch_start_index = batch_num * batch_size
            batch_end_index = min((batch_num + 1) * batch_size, len(training_data))
            batch = training_data[batch_start_index:batch_end_index]
            
            [sentence_batch, sentiment_batch] = zip(*(batch))

            h1_batch = []
            y_batch = []

            for sentence in sentence_batch:
                # Naming our first hidden layer nodes h1
                # Note to future team : Fast Eric made us do this
                h1 = evaluate_layer(hidden_weights, hidden_biases, sentence, activation_function)
                h1_batch.append(h1)
                y = evaluate_layer(output_weights, output_biases, h1, transfer_function)
                y_batch.append(y)

            hidden_layer = Layer(input_batch = sentence_batch, weights = hidden_weights, biases = hidden_biases, activation_derivative = activation_derivative)
            output_layer = Layer(input_batch = h1_batch, weights = output_weights, biases = output_biases, activation_derivative = derivative_cross_entropy_with_softmax)

            layers = [output_layer, hidden_layer]
        
            backprop(sentiment_batch, y_batch, layers)

            print("Epoch #", epoch_num, ", batch #", batch_num, ", cost = ", cross_entropy(training_sentiment[0], y))

        # if sentence_index % 100 == 0:
        #     print("Sentence #", sentence_index + 1, ", cost = ", cross_entropy(training_sentiment[sentence_index], y))


def train_one_sentence(iterations):
    for i in range(0, iterations):
        # Naming our first hidden layer nodes h1
        # Note to future team : Fast Eric made us do this
        h1 = evaluate_layer(hidden_weights, hidden_biases, sentence_input[0], activation_function)

        y = evaluate_layer(output_weights, output_biases, h1, transfer_function)

        hidden_layer = Layer(inputs = sentence_input[0], weights = hidden_weights, biases = hidden_biases, activation_derivative = activation_derivative)
        output_layer = Layer(inputs = h1, weights = output_weights, biases = output_biases, activation_derivative = derivative_cross_entropy_with_softmax)
        layers = [output_layer, hidden_layer]
        backprop(training_sentiment[0], y, layers)

        if i % 100 == 0:
            print("Iteration #", i, ", cost = ", cross_entropy(training_sentiment[0], y))


train_all_sentences()


# Next steps: Mini-batch, then add another hidden layer (h2), then add TensorFlow, then add word2vec.
