import sys
from collections import defaultdict
from math import log, exp

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.


# Remove trace tokens and tags from the treebank as these are not necessary.
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]


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
# for convenience it also stores a long list of single words and returns it
def preprocess_text(sentence_set, vocabulary):
    prep_list = []
    word_list = []  # for list of words
    for sentence in sentence_set:
        word_list.append(start_token)  # for list of words
        for i in xrange(len(sentence)):
            if sentence[i] not in vocabulary:
                sentence[i] = unknown_token
            word_list.append(sentence[i])  # for list of words
        word_list.append(end_token)  # for list of words
        sentence.append(end_token)
        sentence.insert(0, start_token)
        prep_list.append(sentence)

    return prep_list, word_list
class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        
    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """
            
    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        
    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        
    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        
    def Test(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """

def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    
def ComputeAccuracy(test_set, test_set_predicted):
    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """


def main():
    # Remove trace tokens.
    treebank_tagged_sents = TreebankNoTraces()
    # This is the train-test split that we will use.
    training_set = treebank_tagged_sents[:3000]
    test_set = treebank_tagged_sents[3000:]
    
    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    # training_set_prep = PreprocessText(training_set, vocabulary)
    # test_set_prep = PreprocessText(test_set, vocabulary)
    
    """ Print the first sentence of each data set.
    """
    print training_set[0]
    # print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    # print " ".join(untag(test_set_prep[0]))
    #
    #
    # """ Estimate Bigram HMM from the training set, report level of ambiguity.
    # """
    # bigram_hmm = BigramHMM()
    # bigram_hmm.Train(training_set_prep)
    # print "Percent tag ambiguity in training set is %.2f%%." \
    #       %bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    # print "Joint probability of the first sentence is %s." \
    #       %bigram_hmm.JointProbability(training_set_prep[0])
    #
    #
    # """ Implement the most common class baseline. Report accuracy of the predicted tags.
    # """
    # test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    # print "--- Most common class baseline accuracy ---"
    # ComputeAccuracy(test_set_prep, test_set_predicted_baseline)
    #
    #
    # """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    # """
    # test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    # print "--- Bigram HMM accuracy ---"
    # ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)


if __name__ == "__main__": 
    main()