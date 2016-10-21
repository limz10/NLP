import sys
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.


# initialize vocabulary from the training set
def init_vocabulary(word_set):
    diction = defaultdict(int)
    vocab = set()

    for line in word_set:
        for word in line:
            diction[word] += 1

    for element in diction:
        if diction[element] >= 2:
            vocab.add(element)

    vocab.add(unknown_token)
    vocab.add(start_token)
    vocab.add(end_token)

    return vocab


# get the sentence set and known vocabulary
# add start_token and end_token to sentences
# replace unknown words with unknown_token
# return a list of processed sentences
def preprocess_text(sentence_set, vocabulary):
    prep_list = []

    for sentence in sentence_set:
        for i in xrange(len(sentence)):
            if sentence[i] not in vocabulary:
                sentence[i] = unknown_token

        sentence.append(end_token)
        sentence.insert(0, start_token)
        prep_list.append(sentence)

    return prep_list


class BigramLM:
    def __init__(self, vocabulary=set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.log_probs = {}
    """ Implement the functions EstimateBigrams, CheckDistribution, Perplexity and any 
    other function you might need here.
    """


def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]

    vocabulary = init_vocabulary(training_set)

    """ Transform the data sets by eliminating unknown words and adding sentence boundary 
    tokens.
    """
    training_set_prep = preprocess_text(training_set, vocabulary)
    held_out_set_prep = preprocess_text(held_out_set, vocabulary)
    test_set_prep = preprocess_text(test_set, vocabulary)

    """ Print the first sentence of each data set.
    """
    print training_set_prep[0]
    print held_out_set_prep[0]
    print test_set_prep[0]

    """ Estimate a bigram_lm object, check its distribution, compute its perplexity.
    """

    """ Print out perplexity after Laplace smoothing.
    """ 

    """ Print out perplexity after simple linear interpolation (SLI) with lambda = 0.5.
    """ 

    """ Estimate interpolation weights using the deleted interpolation algorithm on the 
    held out set and print out.
    """ 

    """ Print out perplexity after simple linear interpolation (SLI) with the estimated
    interpolation weights.
    """ 

if __name__ == "__main__": 
    main()







    