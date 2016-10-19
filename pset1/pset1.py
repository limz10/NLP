from collections import defaultdict


class FSA:
        def __init__(self, num_states = 0):
                self.num_states = num_states
                self.transitions = defaultdict(list)
                self.final_states = set()

        def addTransition(self, input, curr_state, next_state):
                for i in input:
                        self.transitions[(i, curr_state)] = next_state

        def setFinalState(self, final_state):
                self.final_states.add(final_state)

        def findNextState(self, input, curr_state):
                return self.transitions[(input, curr_state)]

        def checkFinalState(self, curr_state):
                return curr_state in self.final_states


def NDRecognize(input, fsa):
        if len(input) == 0:
                return False
        if len(input) == 1 and input == "0":
                return False

        agenda = [(0, 0)]
        curr_state, index = agenda.pop()

        while True:
                if acceptState(curr_state, fsa, input, index):
                        return True
                else:
                        # print agenda, curr_state, input, index
                        agenda.extend(generateNewStates(curr_state, fsa, input[index], index))
                        # print agenda

                if len(agenda) == 0:
                        return False
                else:
                        curr_state, index = agenda.pop()


def acceptState(curr_state, fsa, input, index):
        if index == len(input) and fsa.checkFinalState(curr_state):
                return True
        else:
                return False


def generateNewStates(curr_state, fsa, input, index):
        to_ret = []

        if (input, curr_state) in fsa.transitions:
                next_state = fsa.findNextState(input, curr_state)
                tape_index = index + 1
                to_ret = [(next_state, tape_index)]

        if ("eps", curr_state) in fsa.transitions:
                # print "EPS True"
                to_ret.append([fsa.findNextState("eps", curr_state), index])

        return to_ret


def concatenate(fsa1, fsa2):
        fsa = FSA(fsa1.num_states + fsa2.num_states)
        fsa.setFinalState(fsa.num_states)
        fsa.transitions = fsa1.transitions
        
        for i in fsa2.transitions.keys():
                # print i
                input2 = [i[0]]
                curr_state2 = i[1]
                next_state2 = fsa2.transitions[i]
                # (input2, curr_state2), next_state2 = i
                curr_state2 += fsa1.num_states
                next_state2 += fsa1.num_states
                # print input2, curr_state2, next_state2
                fsa.addTransition(input2, curr_state2, next_state2)
                # print fsa.transitions.keys()

        # print fsa.transitions

        return fsa


# Building FSA
nonzeroDigits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Months
months = FSA(3)
months.setFinalState(3)
months.addTransition(["0"], 0, 1)
months.addTransition(["1"], 0, 2)
months.addTransition(['eps'], 0, 1)
months.addTransition(nonzeroDigits, 1, 3)
months.addTransition(["0", "1", "2"], 2, 3)


# Days
days = FSA(4)
days.setFinalState(4)
days.addTransition(["0"], 0, 1)
days.addTransition(["1", "2"], 0, 2)
days.addTransition(["3"], 0, 3)
days.addTransition(['eps'], 0, 1)
days.addTransition(nonzeroDigits, 1, 4)
days.addTransition(digits, 2, 4)
days.addTransition(["0", "1"], 3, 4)


# Years
years = FSA(5)
years.setFinalState(5)
years.addTransition(["1"], 0, 1)
years.addTransition(["2"], 0, 2)
years.addTransition(["9"], 1, 3)
years.addTransition(["0"], 2, 3)
years.addTransition(digits, 3, 4)
years.addTransition(digits, 4, 5)

# Separators
seps = FSA(1)
seps.setFinalState(1)
seps.addTransition(["-", " ", "/"], 0, 1)

# Dates
dates = concatenate(months, seps)
# print "CONCATENATED!"
# print dates.transitions.keys()
dates = concatenate(dates, days)
# print dates.transitions.keys()
dates = concatenate(dates, seps)
# print dates.transitions.keys()
dates = concatenate(dates, years)
# print dates.transitions.keys()

# Testing

# for i in dates.transitions.items():
#     print i
# result = NDRecognize("06/04/1989", dates)
# print result


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
TestDates(dates)
