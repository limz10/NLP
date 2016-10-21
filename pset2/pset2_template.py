import sys
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Implement any helper functions here, e.g., for text preprocessing.
"""

def preprocessText():



class BigramLM:
    def __init__(self, vocabulary = set()):
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

    """ Transform the data sets by eliminating unknown words and adding sentence boundary 
    tokens.
    """
    training_set_prep = preprocessText(training_set, vocabulary)
    held_out_set_prep = preprocessText(held_out_set, vocabulary)
    test_set_prep = preprocessText(test_set, vocabulary)

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







    