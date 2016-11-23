import sys, re
import nltk
from nltk.corpus import treebank
from collections import defaultdict
from nltk import induce_pcfg
from nltk.grammar import Nonterminal
from nltk.tree import Tree
from math import exp, pow

unknown_token = "<UNK>"  # unknown word token.

""" Removes all function tags e.g., turns NP-SBJ into NP.
"""         
def RemoveFunctionTags(tree):
    for subtree in tree.subtrees():  # for all nodes of the tree
        # if it's a preterminal node with the label "-NONE-", then skip for now
        if subtree.height() == 2 and subtree.label() == "-NONE-": continue
        nt = subtree.label()  # get the nonterminal that labels the node
        labels = re.split("[-=]", nt)  # try to split the label at "-" or "="
        if len(labels) > 1:  # if the label was split in two e.g., ["NP", "SBJ"]
            subtree.set_label(labels[0])  # only keep the first bit, e.g. "NP"


""" Return true if node is a trace node.
"""         
def IsTraceNode(node):
    # return true if the node is a preterminal node and has the label "-NONE-"
    return node.height() == 2 and len(node) == 1 and node.label() == "-NONE-"


""" Deletes any trace node children and returns true
    if all children were deleted.
"""
def RemoveTraces(node):
    if node.height() == 2:  # if the node is a preterminal node
        return False  # already a preterminal, cannot have a trace node child.
    i = 0
    while i < len(node):  # iterate over the children, node[i]
        # if the child is a trace node or it is a node whose children were deleted
        if IsTraceNode(node[i]) or RemoveTraces(node[i]):
            del node[i]  # then delete the child
        else: i += 1
    return len(node) == 0  # return true if all children were deleted


""" Preprocessing of the Penn treebank.
"""
def TreebankNoTraces():
    tb = []
    for t in treebank.parsed_sents():
        if t.label() != "S": continue
        RemoveFunctionTags(t)
        RemoveTraces(t)
        t.collapse_unary(collapsePOS = True, collapseRoot = True)
        t.chomsky_normal_form()
        tb.append(t)
    return tb


""" Enumerate all preterminal nodes of the tree.
""" 
def PreterminalNodes(tree):
    for subtree in tree.subtrees():
        if subtree.height() == 2:
            yield subtree


""" Print the tree in one line no matter how big it is
    e.g., (VP (VB Book) (NP (DT that) (NN flight)))
"""         
def PrintTree(tree):
    if tree.height() == 2: return "(%s %s)" %(tree.label(), tree[0])
    return "(%s %s)" %(tree.label(), " ".join([PrintTree(x) for x in tree]))


""" Initialize vocabulary from the training set
"""
def init_vocab(training_set):
    vocab = set()
    dictionary = defaultdict(int)
    for sentence in training_set:
        for word in sentence.leaves():
            if word in dictionary:
                vocab.add(word)
            dictionary[word] += 1
    return vocab


""" As usual, build a static vocabulary from the training set,
    treating every word that occurs not more than once as an unknown token.
"""
def PreprocessText(text_set, vocab):
    prep_list = []
    for sent in text_set:
        for NPsubtree in PreterminalNodes(sent):
            if NPsubtree[0] not in vocab:
                NPsubtree[0] = unknown_token
        prep_list.append(sent)
    return prep_list


""" Learning a PCFG from dataset
"""
def learn_PCFG(text_set, start_token):
    s = Nonterminal(start_token)
    production_list = []
    for sent in text_set:
        production_list += sent.productions()

    return induce_pcfg(s, production_list)


class InvertedGrammar:
    def __init__(self, pcfg):
        self._pcfg = pcfg
        self._r2l = defaultdict(list)  # maps RHSs to list of LHSs
        self._r2l_lex = defaultdict(list)  # maps lexical items to list of LHSs
        self.BuildIndex()  # populates self._r2l and self._r2l_lex according to pcfg


    def PrintIndex(self, filename):
        f = open(filename, "w")
        for rhs, prods in self._r2l.iteritems():
            f.write("%s\n" %str(rhs))
            for prod in prods:
                f.write("\t%s\n" %str(prod))
            f.write("---\n")
        for rhs, prods in self._r2l_lex.iteritems():
            f.write("%s\n" %str(rhs))
            for prod in prods:
                f.write("\t%s\n" %str(prod))
            f.write("---\n")
        f.close()

    def BuildIndex(self):
        """ Build an inverted index of your grammar that maps right hand sides of all
        productions to their left hands sides.
        """
        for production in self._pcfg.productions():
            if production.is_lexical():
                self._r2l_lex[production.rhs()].append(production)
            else:
                self._r2l[production.rhs()].append(production)

        self.PrintIndex("index")

    def Parse(self, sent):
        """ Implement the CKY algorithm for PCFGs,
            populating the dynamic programming table with log probabilities of
            every constituent spanning a sub-span of a given
            test sentence (i, j) and storing the appropriate back-pointers.
        """
        table = defaultdict(dict)
        backpointers = defaultdict(dict)
        for j in xrange(1, len(sent) + 1):
            for A in self._r2l_lex[tuple([sent[j - 1]])]:
                table[(j - 1, j)][A.lhs()] = A.logprob()
            if j >= 2:
                for i in reversed(xrange(j - 1)):
                    for k in xrange(i + 1, j):
                        for B in table[(i, k)]:
                            for C in table[(k, j)]:
                                for A in self._r2l[(B, C)]:
                                    temp = A.logprob() + table[(i, k)][B] + \
                                           table[(k, j)][C]
                                    if A.lhs() not in table[(i, j)]:
                                        table[(i, j)][A.lhs()] = temp
                                        backpointers[(i, j)][A.lhs()] = (k, B, C)
                                    elif table[(i, j)][A.lhs()] < temp:
                                        table[(i, j)][A.lhs()] = temp
                                        backpointers[(i, j)][A.lhs()] = (k, B, C)

        return table, backpointers

    @staticmethod
    def BuildTree(cky_table, sent):
        """ Build a tree by following the back-pointers starting from the largest span
            (0, len(sent)) and recursing from larger spans (i, j) to smaller sub-spans
            (i, k), (k, j) and eventually bottoming out at the preterminal level (i, i+1).
        """
        if Nonterminal('S') not in cky_table[(0, len(sent))]:
            return None
        else:
            return InvertedGrammar.recursive_build(cky_table, sent, Nonterminal("S"), 0, len(sent))

    @staticmethod
    def recursive_build(cky_back, sent, nt, i, j):
        if j - i == 1:
            TreeOut = Tree(nt.symbol(), [sent[i]])
        else:
            (k, B, C) = cky_back[(i, j)][nt]
            TreeOut = Tree(nt.symbol(), [
                InvertedGrammar.recursive_build(cky_back, sent, B, i, k),
                InvertedGrammar.recursive_build(cky_back, sent, C, k, j)])

        return TreeOut


def bucketing(test_set_prep):
    bucket1 = []
    bucket2 = []
    bucket3 = []
    bucket4 = []
    bucket5 = []

    for sent in test_set_prep:
        if 0 < len(sent.leaves()) < 10:
            bucket1.append(sent)
        elif 10 <= len(sent.leaves()) < 20:
            bucket2.append(sent)
        elif 20 <= len(sent.leaves()) < 30:
            bucket3.append(sent)
        elif 30 <= len(sent.leaves()) < 40:
            bucket4.append(sent)
        elif len(sent.leaves()) >= 40:
            bucket5.append(sent)

    return bucket1, bucket2, bucket3, bucket4, bucket5


def main():
    treebank_parsed_sents = TreebankNoTraces()
    training_set = treebank_parsed_sents[:3000]
    test_set = treebank_parsed_sents[3000:]

    """ Transform the data sets by eliminating unknown words.
    """
    vocabulary = init_vocab(training_set)
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    print PrintTree(training_set_prep[0])
    print PrintTree(test_set_prep[0])

    """ Implement your solutions to problems 2-4.
    """

    """ Training a PCFG
    """
    pcfg = learn_PCFG(training_set_prep, "S")
    NP_dict = {}
    for production in pcfg.productions():
        if str(production.lhs()) == "NP":
            NP_dict[production] = production.prob()

    print "Total number for NP nonterminal: ", len(NP_dict), " \n"
    print "The most probable 10 productions for the NP nonterminal: \n"
    print sorted(NP_dict, key=NP_dict.get, reverse=True)[:9], " \n"


    """ Testing: Implement the probabilistic CKY algorithm for parsing a test
        sentence using your learned PCFG.
    """
    ig = InvertedGrammar(pcfg)
    sample_sentence = ['Terms', 'were', "n't", 'disclosed', '.']
    table, tree = ig.Parse(sample_sentence)
    print 'The log probability of the 5-token sentence: ', \
        table[(0, len(sample_sentence))][Nonterminal('S')]
    print 'The parse tree for the 5-token sentence:\n', \
        ig.BuildTree(tree, sample_sentence)

    """ Bucketing
    """
    bucket1, bucket2, bucket3, bucket4, bucket5 = bucketing(test_set_prep)
    print "Number of sentences in each bucket: ", \
        len(bucket1), len(bucket2), len(bucket3), len(bucket4), len(bucket5)

    test_bucket = bucket2

    test_file = open('test_2', "w")
    gold_file = open('gold_2', "w")
    count = 0
    for sent in test_bucket:
        count += 1
        temp_tree = ig.BuildTree(ig.Parse(sent.leaves())[1], sent.leaves())
        sent.un_chomsky_normal_form()

        if temp_tree is None:
            test_file.write('\n')
        else:
            temp_tree.un_chomsky_normal_form()
            test_file.write(PrintTree(temp_tree) + '\n')
        gold_file.write(PrintTree(sent) + '\n')

        print count
    test_file.close()
    gold_file.close()


if __name__ == "__main__": 
    main()






