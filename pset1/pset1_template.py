from collections import defaultdict


class FSA:
    def __init__(self, num_states = 0):
        self.num_states = num_states
        self.transitions = defaultdict(list)
        self.final_states = set()

    def addTransition(self, input, curr_state, next_state):
        self.transitions[(input, curr_state)] = next_state

    def addTransList(self, input_list, curr_state, next_state):
        for i in input_list:
            self.addTransition(i, curr_state, next_state)

    def setFinalState(self, final_state):
            self.final_states.add(final_state)

    def findNextState(self, input, curr_state):
            return self.transitions[(input, curr_state)]

    def checkFinalState(self, curr_state):
            return curr_state in self.final_states


def NDRecognize(input, fsa):
	if len(input) == 0:
		return False
	agenda = [(0, 0)]
	current_state, index = agenda.pop()

	while True:
		if AcceptState(input, fsa, current_state, index):
			return True
		else:
			agenda.extend(
				GenerateNewStates(input[index], current_state, index, fsa))
		if len(agenda) == 0:
			return False
		else:
			current_state, index = agenda.pop()


def GenerateNewStates(input_symbol, current_state, index, fsa):
	if (input_symbol, current_state) in fsa.transitions:
		pot_next = fsa.FindNextState(input_symbol, current_state)
		pot_idx = index + 1
		result = [(pot_next, pot_idx)]
	else:
		result = []
	if ('eps', current_state) in fsa.transitions:
		result.append((fsa.FindNextState('eps', current_state), index))
	return result


def AcceptState(input, fsa, current_state, index):
	if index == len(input) and fsa.CheckFinalState(current_state):
		return True

def concatenate(fsa1, fsa2):
    """ TODO: Implement Concatenate so that if w1 is a string that fsa1 accepts and w2 is
    a string that fsa2 accepts this function should return an fsa that accepts w1w2.
    """


""" Below are some test cases. Include the output of this in your write-up and provide
explanations.
"""


def TestComponents(months, days, years, seps):
    print "\nTest Months FSA"
    for input in ["", "0", "1", "9", "10", "11", "12", "13"]:
        print "'%s'\t%s" %(input, NDRecognize(input, months))
    print "\nTest Days FSA"
    for input in ["", "0", "1", "9", "10", "11", "21", "31", "32"]:
        print "'%s'\t%s" %(input, NDRecognize(input, days))
    print "\nTest Years FSA"
    for input in ["", "1899", "1900", "1901", "1999", "2000", "2001", "2099", "2100"]:
        print "'%s'\t%s" %(input, NDRecognize(input, years))
    print "\nTest Separators FSA"
    for input in ["", ",", " ", "-", "/", "//", ":"]:
        print "'%s'\t%s" %(input, NDRecognize(input, seps))


def TestDates(dates):
    print "\nTest Date Expressions FSA"
    for input in ["", "12 31 2000", "12/31/2000", "12-31-2000", "12:31:2000",
                  "1 2 2000", "1/2/2000", "1-2-2000", "1:2:2000",
                  "00-31-2000", "12-00-2000", "12-31-0000",
                  "12-32-1987", "13-31-1987", "12-31-2150"]:
        print "'%s'\t%s" %(input, NDRecognize(input, dates))

# TestComponents(months, days, years, seps)
# TestDates(dates)

