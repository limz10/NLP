from collections import defaultdict

class FSA:
    def __init__(self, num_states = 0):
        self.num_states = num_states
        self.transitions = defaultdict(list)
        self.final_states = set()
    """ TODO: Add methods for adding transitions, setting final states, looking up next
    state in the state transitions table, checking whether or not a state is a final 
    (accepting) state. 
    """

def NDRecognize(input, fsa):
    """ TODO: Implement ND-RECOGNIZE from SLP Figure 2.19, return true or false based on 
    whether or not the nondeterministic fsa object accepts or rejects the input string.
    """
    
def Concatenate(fsa1, fsa2):
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
		
TestComponents(months, days, years, seps)
TestDates(dates)







