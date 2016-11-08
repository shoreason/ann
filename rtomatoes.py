# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
# from KaggleWord2VecUtility import KaggleWord2VecUtility

# read labeled training data
import pandas as pd
train = pd.read_csv("data/train.tsv", header=0, delimiter="\t", quoting=3)

# nltk.download()

# 156,060 rows, 4 columns
print("number of rows, columns is : " + str(train.shape))

# column values ['PhraseId' 'SentenceId' 'Phrase' 'Sentiment']
print("column names " + str(train.columns.values))

num_phrases = train["Phrase"].size

clean_train_phrases = []

# 0 - negative, 1 - somewhat negative, 2- neutral,
# 3 - somewhat positive, 4 - positivecear

for i in range(0, 10):
    if (train["Sentiment"][i] == 1):
       print(train["PhraseId"][i])
       print(train["SentenceId"][i])
       print(train["Phrase"][i])
       print(train["Sentiment"][i])


# tidy up phrases
def phrase_to_wordlist(raw_phrase, remove_stopwords=False):
    # remove html elements
    phrase_text = BeautifulSoup(raw_phrase, "html.parser").get_text()

    # remove everything but letters
    # letters_only = re.sub("[^a-zA-Z]", " ", phrase_text)

    # make words lowercase
    words = letters_only.lower().split()

    # setup stopwords and remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(" ".join(words))

# for i in range(0, num_phrases):
    # if((i + 1) % 1000 == 0):
        # print("Phrase %d of %d\n" % (i + 1, num_phrases))
    # print(phrase_to_words(train["Phrase"][i]))
    # clean_train_phrases.append(phrase_to_wordlist(train["Phrase"][i]))
