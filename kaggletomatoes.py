import pandas as pd
from collections import Counter

PAD_INDEX = 0
UNKNOWN_INDEX = 1

train = pd.read_csv("data/train.tsv", header=0, delimiter="\t", quoting=3)

num_phrases = train["PhraseId"].size

sentences = []
last_sentence_id = 0
for i in range(0, num_phrases):
    sentence_id = train["SentenceId"][i]
    if sentence_id != last_sentence_id:
        sentences.append(train["Phrase"][i].split())
        last_sentence_id = sentence_id

print(sentences[0:2])

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
    else
        return UNKNOWN_INDEX
    # return lookup_table[word] if word in lookup_table else UNKNOWN_INDEX

sentence_input = []
for sentence in sentences:
    numeric_words = list(map(lookup_word, sentence))
    numeric_words += [PAD_INDEX] * (sentence_max - len(numeric_words))
    sentence_input.append(numeric_words)

print(sentence_input[0:2])
