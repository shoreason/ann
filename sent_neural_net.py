import pandas as pd
import tensorflow as tf
from collections import Counter
import random
import math
import nltk
from nltk.corpus import stopwords
import re
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

#model = {"movie": list(range(0, 5)), "film": list(range(0, 5)), "series": list(range(1, 6))}

#print(model.doesnt_match("man woman child kitchen".split()))

PAD_INDEX = 0
UNKNOWN_INDEX = 1

hidden_layer_size = 800
learning_rate = 0.001
iteration_count = 3000
batch_size = 50
num_epochs = 10
embedding_size = 300


train = pd.read_csv("data/train.tsv", header=0, delimiter="\t", quoting=3)

num_phrases = train["PhraseId"].size
clean_train_phrases = []
training_sentiment = []


# Fast Fola suggested this naming convention
def hot_vectorize(sentiment):
    one_hot_vector = [0,0,0,0,0]
    one_hot_vector[sentiment-1]=1
    return one_hot_vector

def phrase_to_wordlist(raw_phrase, remove_stopwords=False):
    # make words lowercase
    words = raw_phrase.lower().split()

    # setup stopwords and remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    pattern = re.compile("^[\w]+$")
    words = [w for w in words if pattern.match(w) ]
    words = [w for w in words if len(w)>1 ]

    return(" ".join(words))


sentences = []
last_sentence_id = 0
for i in range(0, num_phrases):
    sentence_id = train["SentenceId"][i]
    if sentence_id != last_sentence_id:
        sentence = phrase_to_wordlist(train["Phrase"][i], remove_stopwords=False)
        sentences.append(sentence.split())
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
print("Counter length: " + str(len(counter)))
print(counter.most_common(10))

i = 2
lookup_table = {}
index_to_word_lookup_table = {PAD_INDEX: "<pad>", UNKNOWN_INDEX: "<unknown>"}
for word, _ in counter.most_common(18000):
    # if counter[word] > 1:
    lookup_table[word] = i
    index_to_word_lookup_table[i] = word
    i += 1

def lookup_word(word):
    if word in lookup_table:
        return lookup_table[word]
    else:
        return UNKNOWN_INDEX
    # return lookup_table[word] if word in lookup_table else UNKNOWN_INDEX

def lookup_word2vec(word):
    if word in model:
        return model[word]
    else:
        return [0] * embedding_size

sentence_input = []
for sentence in sentences:
    # numeric_words = list(map(lookup_word, sentence))
    # numeric_words += [PAD_INDEX] * (sentence_max - len(numeric_words))
    numeric_words = list(map(lookup_word2vec, sentence))
    numeric_words += [([0] * embedding_size) for _ in range(0, sentence_max - len(numeric_words))]
    sentence_input.append(numeric_words)

def lookup_index(index):
    return index_to_word_lookup_table[index]

print("First sentence: ", sentence_input[0])

# images going into input layer (input Layer images)
# x = tf.placeholder(tf.float32, [None, sentence_max])
# W = tf.Variable(tf.truncated_normal([sentence_max, hidden_layer_size], stddev=0.1), name="W")
x = tf.placeholder(tf.float32, [None, sentence_max, embedding_size])
x_reshaped = tf.reshape(x, [-1, sentence_max * embedding_size])
W = tf.Variable(tf.truncated_normal([sentence_max * embedding_size, hidden_layer_size], stddev=0.1), name="W")
b = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name="b")

# Hidden layer
h1 = tf.nn.sigmoid(tf.matmul(x_reshaped, W) + b, name = "h1")
W_h1 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], stddev=0.1), name="W_h1")
b_h1 = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name="b_h1")

h2 = tf.nn.sigmoid(tf.matmul(h1, W_h1) + b_h1, name = "h2")
W_h2 = tf.Variable(tf.truncated_normal([hidden_layer_size, 5], stddev=0.1), name="W_h2")
b_h2 = tf.Variable(tf.truncated_normal([5], stddev=0.1), name="b_h2")

# Actual output
y = tf.nn.softmax(tf.matmul(h2, W_h2) + b_h2, name="y")

# Expected output
y_ = tf.placeholder(tf.float32, [None, 5])

# Cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create session
sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

training_sentences = sentence_input[:-100]
testing_sentences = sentence_input[-100:]

training_data = list(zip(training_sentences, training_sentiment[:-100]))
for epoch_num in range(0, num_epochs):
    random.shuffle(training_data)
    num_batches = int(math.ceil(len(training_data) / batch_size))
    print("len(training_data) = ", len(training_data), "; batch_size = ", batch_size, "; num_batches = ", num_batches)
    for batch_num in range(0, num_batches):
        batch_start_index = batch_num * batch_size
        batch_end_index = min((batch_num + 1) * batch_size, len(training_data))
        batch = training_data[batch_start_index:batch_end_index]

        [sentence_batch, sentiment_batch] = zip(*(batch))
        # if batch_num == 0:
        #     for sentence in sentence_batch[0:2]:
        #         print(list(map(lookup_index, sentence)))
        #     print(sentiment_batch[0:2])

        # print(sentence_batch[0:2])

        sess.run(train_step, feed_dict={x: sentence_batch, y_: sentiment_batch})
        print(sess.run(accuracy, feed_dict={x: sentence_batch, y_: sentiment_batch}))

print("==========================================")
print(sess.run(accuracy, feed_dict={x: testing_sentences, y_: training_sentiment[-100:]}))
