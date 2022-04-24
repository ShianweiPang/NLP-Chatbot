import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# a package with a pre-trained tokenizer
# nltk.download('punkt')

stemmer = PorterStemmer()
def tokenize(sentence):
    # You might get an error running for the first time
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenized, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenized = [stem(word) for word in tokenized]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized: 
            bag[idx] +=1

    return bag

    
