# COMP 150 NLP
# Fall 2016
# Problem Sets 2
# Language Modeling


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


# this function is basically isclose() from the Math library in Python 3
def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class BigramLM:
    def __init__(self, vocabulary=set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(lambda: 0.0)
        self.bigram_counts = defaultdict(lambda: 0.0)
        self.bigram_prob = defaultdict(float)
        self.bigram_dict = defaultdict(set)
        self.training_set_size = int()
        self.lambda1 = float()
        self.lambda2 = float()

    # get both unigram_counts and bigram_counts.
    # generate bigram_dict that contains all known bigrams
    def get_counts(self, word_set):
        self.training_set_size = len(word_set)
        for i in xrange(len(word_set) - 1):
            self.unigram_counts[word_set[i]] += 1
            self.bigram_counts[(word_set[i], word_set[i + 1])] += 1
            self.bigram_dict[word_set[i]].add((word_set[i], word_set[i + 1]))
        self.unigram_counts[word_set[-1]] += 1
        self.unigram_counts[end_token] = 1
        # get rid of end_token bigrams
        del self.bigram_dict[end_token]
        del self.bigram_counts[end_token, start_token]

    # this function returns the Laplacian smoothed probability of bigram (a, b)
    def laplace_smoothing(self, a, b):
        return (float(self.bigram_counts[(a, b)]) + 1) / \
            (self.unigram_counts[a] + len(self.vocabulary))

    # this function calculates the probability for known bigrams
    def estimate_bigrams(self, smoothing, *argv):
        self.bigram_prob.clear()
        if smoothing == "no smoothing":
            for (a, b) in self.bigram_counts:
                self.bigram_prob[(a, b)] = float(self.bigram_counts[(a, b)]) \
                                           / self.unigram_counts[a]

        # "add one" smoothing
        if smoothing == "Laplace smoothing":
            for (a, b) in self.bigram_counts:
                self.bigram_prob[(a, b)] = \
                    (float(self.bigram_counts[(a, b)]) + 1) / \
                    (self.unigram_counts[a] + len(self.vocabulary))

        # for both linear and deleted interpolation, use weight 0.5 to
        # calculate the probability for every known bigram
        if smoothing == 'SLI':
            for (a, b) in self.bigram_counts:
                self.bigram_prob[(a, b)] = 0.5 * \
                    (float(self.bigram_counts[(a, b)]) / self.unigram_counts[a]
                     + float(self.unigram_counts[b]) / self.training_set_size)

        if smoothing == 'deleted interpolation':
            if len(argv) == 0:
                print "held out corpus missing!"
                exit(1)

            # Get counts for held_out_set
            held_out = argv[0]
            held_out_unigram_counts = defaultdict(int)
            held_out_bigram_counts = defaultdict(int)
            for i in range(len(held_out) - 1):
                held_out_unigram_counts[held_out[i]] += 1
                held_out_bigram_counts[(held_out[i], held_out[i + 1])] += 1
            held_out_unigram_counts[held_out[-1]] += 1
            # calculate lambdas
            lambda1 = 0.0
            lambda2 = 0.0
            for (a, b) in held_out_bigram_counts:
                if held_out_unigram_counts[a] == 1:
                    temp2 = 0
                else:
                    temp2 = float((held_out_bigram_counts[(a, b)] - 1)) / \
                            (held_out_unigram_counts[a] - 1)
                temp1 = float((held_out_unigram_counts[b] - 1)) / \
                        (len(held_out) - 1)
                if temp1 > temp2:
                    lambda1 += held_out_bigram_counts[(a, b)]
                else:
                    lambda2 += held_out_bigram_counts[(a, b)]
            temp = lambda1 + lambda2
            self.lambda1 = float(lambda1) / temp
            self.lambda2 = float(lambda2) / temp
            # use the new lambda we get to calculate probabilities
            for (a, b) in self.bigram_counts:
                self.bigram_prob[(a, b)] = \
                    self.lambda2 * float(self.bigram_counts[(a, b)]) / \
                    self.unigram_counts[a] \
                    + self.lambda1 * float(self.unigram_counts[b]) / \
                    self.training_set_size

    # check if the sum of bigram probs is 1
    # end_token is eliminated because there is no doubt that
    # it must be followed by a start_token
    def check_distribution(self):
        flag = 1
        for word in self.vocabulary:
            if word != end_token:
                probs_sum = 0.0
                for bigram in self.bigram_dict[word]:
                    if bigram in self.bigram_prob:
                        probs_sum += self.bigram_prob[bigram]
                    else:
                        print bigram, "Not in bigram_prob!\n"
                        return

                flag &= isclose(probs_sum, 1.0)

        # prompt invalid distribution if any of the flag is not 1
        if flag == 1:
            return "Valid distribution!"
        else:
            return "Invalid distribution!"

    # Perplexity calculation
    def perplexity(self, data, smoothing):
        pp = 0.0
        for i in range(len(data) - 1):
            if smoothing == 'no smoothing':
                if (data[i], data[i + 1]) != (end_token, start_token):
                    if (data[i], data[i + 1]) in self.bigram_prob:
                        pp += log(self.bigram_prob[(data[i], data[i + 1])])
                    else:
                        print data[i], data[i+1]
                        pp = float('inf')
                        return pp

            if smoothing == 'Laplace smoothing':
                if (data[i], data[i + 1]) != (end_token, start_token):
                    if (data[i], data[i + 1]) in self.bigram_prob:
                        pp += log(self.laplace_smoothing(data[i], data[i + 1]))
                    else:
                        pp += log(1.0 / (self.unigram_counts[data[i]] +
                                         len(self.vocabulary)))

            if smoothing == 'SLI':
                if (data[i], data[i + 1]) != (end_token, start_token):
                    if (data[i], data[i + 1]) in self.bigram_prob:
                        pp += log(self.bigram_prob[(data[i], data[i + 1])])
                    else:
                        if data[i + 1] in self.unigram_counts:
                            pp += log(0.5 * float(
                                self.unigram_counts[data[i + 1]]) / (
                                        self.training_set_size))
                        else:
                            print "SLI word OOV!"
                            return

            if smoothing == 'deleted interpolation':
                if (data[i], data[i + 1]) != (end_token, start_token):
                    if (data[i], data[i + 1]) in self.bigram_prob:
                        pp += log(self.bigram_prob[(data[i], data[i + 1])])
                    else:
                        if data[i + 1] in self.unigram_counts:
                            pp += log(self.lambda1 * float(
                                self.unigram_counts[
                                    data[i + 1]]) / self.training_set_size)
                        else:
                            print "Deleted interpolation word OOV!"
                            return

        pp = exp(-(pp / len(data)))
        return pp


def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]

    vocabulary = init_vocabulary(training_set)

    # Transform the data sets by eliminating unknown words
    # and adding sentence boundary tokens.

    training_set_prep, training_word = preprocess_text(training_set, vocabulary)
    held_out_set_prep, held_out_word = preprocess_text(held_out_set, vocabulary)
    test_set_prep, test_word = preprocess_text(test_set, vocabulary)

    # Print the first sentence of each data set.

    print " ".join(training_set_prep[0])
    print " ".join(held_out_set_prep[0])
    print " ".join(test_set_prep[0])

    # Estimate a bigram_lm object, check its distribution,
    # compute its perplexity.
    lm = BigramLM(vocabulary)
    lm.get_counts(training_word)

    lm.estimate_bigrams("no smoothing")

    print "Distribution Check: ", lm.check_distribution()
    print "Perplexity without smoothing: ", \
        lm.perplexity(test_word, "no smoothing")

    # Print out perplexity after Laplace smoothing.
    lm.estimate_bigrams("Laplace smoothing")
    print "Perplexity with Laplace smoothing: ", \
        lm.perplexity(test_word, "Laplace smoothing")

    # Print out perplexity after simple linear interpolation (SLI)
    # with lambda = 0.5.
    lm.estimate_bigrams("SLI")
    print "Perplexity with linear interpolation: ",\
        lm.perplexity(test_word, "SLI")

    # Estimate interpolation weights using the deleted interpolation algorithm
    # on the held out set and print out.
    lm.estimate_bigrams("deleted interpolation", held_out_word)
    print "lambda 1 = ", lm.lambda1
    print "lambda 2 = ", lm.lambda2

    # Print out perplexity after simple linear interpolation (SLI)
    # with the estimated interpolation weights.
    print "Perplexity with deleted interpolation: ", \
        lm.perplexity(test_word, "deleted interpolation")


if __name__ == "__main__": 
    main()
