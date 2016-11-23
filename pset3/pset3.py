import sys
from collections import defaultdict
from math import log, exp
import nltk
from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 


unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.


# Remove trace tokens and tags from the treebank as these are not necessary.
def treebank_no_traces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]


# initialize vocabulary from the training set
def init_vocabulary(word_set):
    diction = defaultdict(int)
    vocab = set()

    for line in word_set:
        for tuple in line:
            if tuple[0] in diction:
                vocab.add(tuple[0])
            diction[tuple[0]] += 1

    vocab.add(unknown_token)
    vocab.add(start_token)
    vocab.add(end_token)

    return vocab


def preprocess_text(sentence_set, vocabulary):
    """
    get the sentence set and known vocabulary
    add start_token and end_token to sentences
    replace unknown words with unknown_token
    return a list of processed sentences
    """
    prep_list = []
    for sentence in sentence_set:
        for i in xrange(len(sentence)):
            if sentence[i][0] not in vocabulary:
                temp = sentence[i][1]
                sentence[i] = unknown_token, temp

        sentence.append((end_token, end_token))
        sentence.insert(0, (start_token, start_token))
        prep_list.append(sentence)

    return prep_list


class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(set)
        self.training_set_size = 0
        
    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """

        word_count = defaultdict(int)
        tag_count = defaultdict(int)
        bigram_count = defaultdict(int)

        # record counts
        for line in training_set:
            self.training_set_size += len(line)
            for i in xrange(len(line)-1):
                word_count[line[i]] += 1
                tag_count[line[i][1]] += 1
                bigram_count[(line[i][1], line[i+1][1])] += 1
                self.dictionary[line[i][0]].add(line[i][1])
            self.dictionary[end_token].add(end_token)

        # estimate bigram transition probabilities
        for tuple in bigram_count:
            self.transitions[tuple] = \
                log(float(bigram_count[tuple]) / tag_count[tuple[0]])

        # estimate emission probabilities
        for word in word_count:
            self.emissions[word] = \
                log(float(word_count[word])/tag_count[word[1]])

    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that
            have more than one tag according to self.dictionary.
        """
        ambiguous_tag_count = 0
        token_count = 0
        for line in data_set:
            for tuple in line:
                token_count += 1
                if len(self.dictionary[tuple[0]]) > 1:
                    ambiguous_tag_count += 1

        return float(ambiguous_tag_count) / token_count * 100  # in percentage
        
    def JointProbability(self, sent):
        """ Compute the joint probability of the words
            and tags of a tagged sentence.
        """
        p = 0.0
        for i in xrange(len(sent) - 1):
            p += self.emissions[sent[i]] + \
                 self.transitions[(sent[i][1], sent[i+1][1])]

        return exp(p)
        
    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely
            tag sequence given the sentence.
        """
        viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)
        tag_path = []
        tag_sent = []

        # initialization step
        for state in self.dictionary[sent[1][0]]:
            if (start_token, state) in self.transitions:
                viterbi[state][1] = self.transitions[(start_token, state)] \
                                      + self.emissions[(sent[1][0], state)]
            else:
                viterbi[state][1] = -float('inf')
            backpointer[state][1] = start_token

        # recursion step
        for i in xrange(2, len(sent)):
            for state in self.dictionary[sent[i][0]]:
                max_value = -float('inf')
                max_location = []
                for prev_state in self.dictionary[sent[i-1][0]]:
                    if (prev_state, state) in self.transitions:
                        temp = viterbi[prev_state][i-1] \
                               + self.transitions[(prev_state, state)]
                    else:
                        temp = -float('inf')
                    if temp >= max_value:
                        max_value = temp
                        max_location = prev_state
                viterbi[state][i] = max_value \
                                         + self.emissions[(sent[i][0], state)]
                backpointer[state][i] = max_location

        temp_path = end_token
        tag_path.append(temp_path)

        for i in xrange(1, len(sent)):
            temp_path = backpointer[temp_path][len(sent)-i]
            tag_path.append(temp_path)
        for tuple in sent:
            tag_sent.append((tuple[0], tag_path.pop()))

        return tag_sent

    def Test(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence
            for every sentence. Return a re-tagged test_set.
        """
        re_tagged = []
        for sent in test_set:
            re_tagged.append(self.Viterbi(sent))

        return re_tagged


def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging.
        Return the test set tagged according to this baseline.
    """
    dictionary = defaultdict(dict)
    mcc_dict = defaultdict()
    test_set_tagged = []

    # learning from the training set, record the frequency
    for line in training_set:
        for tuple in line:
            if tuple[0] not in dictionary:
                dictionary[tuple[0]] = defaultdict(int)
            dictionary[tuple[0]][tuple[1]] += 1

    for item in dictionary:
        freq = 0
        item_key = "NN"  # set default tag to NOUN
        for key in dictionary[item]:
            if dictionary[item][key] > freq:
                freq = dictionary[item][key]
                item_key = key
        mcc_dict[item] = item_key

    for line in test_set:
        temp_line = []
        for i in xrange(len(line)):
            if line[i][0] in dictionary:
                temp_line.append((line[i][0], mcc_dict[line[i][0]]))
            else:
                print "Error!"
                return []

        test_set_tagged.append(temp_line)

    return test_set_tagged


def ComputeAccuracy(test_set, test_set_predicted):
    """ Using the gold standard tags in test_set,
        compute the sentence and tagging accuracy of test_set_predicted.
    """
    if len(test_set) != len(test_set_predicted):
        print "Unequal size! Tagging error!"
        return 0

    word_count = 0
    mistake = 0
    bad_line = 0
    for i in xrange(len(test_set)):
        if test_set[i] == test_set_predicted[i]:
            word_count += len(test_set[i])
        else:
            bad_line += 1
            word_count += len(test_set[i])
            for j in xrange(len(test_set[i])):
                if test_set[i][j] != test_set_predicted[i][j]:
                    mistake += 1

    tagging_accuracy = (1- float(mistake)/(word_count - 2*len(test_set))) * 100
    sentence_accuracy = (1- float(bad_line)/len(test_set)) * 100

    print "Tagging accuracy: %.2f%%." % tagging_accuracy
    print "Sentence accuracy: %.2f%%." % sentence_accuracy


def confusion_matrix(test_set, test_set_predicted):
    confusionMatrix = defaultdict(lambda: defaultdict(int))
    for i in xrange(len(test_set)):
        for j in xrange(1, len(test_set[i]) - 1):
            if test_set_predicted[i][j][1] != test_set[i][j][1]:
                confusionMatrix[test_set[i][j][1]][test_set_predicted[i][j][1]] += 1

    return confusionMatrix


def most_confused(matrix):
    token = []
    error_count = 0
    correct = []

    for real_key in matrix:
        count = max(matrix[real_key].values())
        if count >= error_count:
            error_count = count
            token = max(matrix[real_key], key=matrix[real_key].get)
            correct = real_key

    return token, correct, error_count


def main():
    # Remove trace tokens.
    treebank_tagged_sents = treebank_no_traces()
    # This is the train-test split that we will use.
    training_set = treebank_tagged_sents[:3000]
    test_set = treebank_tagged_sents[3000:]

    vocabulary = init_vocabulary(training_set)

    # Transform the data sets by eliminating unknown words
    # and adding sentence boundary tokens.
    training_set_prep = preprocess_text(training_set, vocabulary)
    test_set_prep = preprocess_text(test_set, vocabulary)

    # Print the first sentence of each data set.
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))

    """ Estimate Bigram HMM from the training set, report level of ambiguity.
    """
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)
    print "Percent tag ambiguity in training set is %.2f%%." \
          % bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    print "Joint probability of the first sentence is %s." \
          % bigram_hmm.JointProbability(training_set_prep[0])

    # for sanity check only
    # s = 0.0
    # for line in training_set_prep:
    #     s += bigram_hmm.JointProbability(line)
    # print s

    """ Implement the most common class baseline.
        Report accuracy of the predicted tags.
    """
    test_set_predicted_baseline = \
        MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)

    """ Use the Bigram HMM to predict tags for the test set.
        Report accuracy of the predicted tags.
    """
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)

    """ Extra Credit: Confusion Matrix
        Reports the most confused token and tag
    """
    confusion = \
        confusion_matrix(test_set_prep, test_set_predicted_bigram_hmm)
    tag, real_tag, error_count = most_confused(confusion)
    print "Most confused token: ", tag
    print "True tag should be: ", real_tag
    print "Confused by ", error_count, " times"

    print nltk.help.upenn_tagset("JJ")
    print nltk.help.upenn_tagset("NN")

if __name__ == "__main__": 
    main()
