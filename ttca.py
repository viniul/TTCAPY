import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
from collections import OrderedDict
import timeit
import csv
import sys

class ODict(OrderedDict):
	''' Inherit from OrderedDict, but add the function get_first '''
	def get_first(self):
		''' Return the first element of the OrderedDict'''
		return next(iter(self))


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

	
class Ttca(): 
	def ttca_ordered_set(prefmatrix,show_progress=True,progress_function=update_progress): 
		''' Take prefmatrix, output which player gets which house''' 
		'''Prefmatrix: i-th row, j-th column: player i ranks player M[i,j] on the j-th place'''
		''' whereby M[i,j] \in {1,...,k} \forall i,j, i.e. Players are denoted with 1...keys''' 
		''' Return: A dictionary, whereby dict[i] is the house that player i gets'''
		prefdict = {} # Prefdict[i] = Preference list of player i as an OrderedDict
		number_of_players = prefmatrix.shape[0]
		for i in range(1,number_of_players+1): # Foreach player
			tmpdict = ODict() # Create an OrderedDict
			for j in range(number_of_players): # 
				tmpdict[prefmatrix[i-1,j]] = None # Add the rank as the j-th keys, i.e. players are inserted in the dict 
				# in the same order as player i ranks them
			prefdict[i] = tmpdict 
		allocation = {} # Allocation dict, player i gets house of allocation[i]
		total_number_of_players = prefmatrix.shape[0]
		print("here")
		while number_of_players>1:
			G = nx.DiGraph()
			# Create the graph: Each player "points" to the player he likes best out of the remaining players
			edges_list = map(lambda i: (i,prefdict[i].get_first()),prefdict.keys())
			G.add_edges_from(edges_list)
			# Find all the cycles in the graph, i.e. all the players that would be better off by trading
			cycles = nx.simple_cycles(G)
			cycles = list(cycles)
			#For each cycle, save the allocation and delete the player.
			for c in cycles: 
				for p in c:
					allocation[p] = prefdict[p].get_first()
			for c in cycles:
				for p in c:
					for d in prefdict.values():
						del d[p]
					del prefdict[p]
					number_of_players -= 1
			if show_progress==True: 
				progress_function(1-(float(number_of_players)/total_number_of_players))
		return allocation

'''	
def ttca(prefmatrix): 
	# Take prefmatrix, output which player gets which house
	# i-th rows, j-th column: player i ranks player M[i,j] on the j-th place
	t = 0
	# Conver the prefmatrix into a list of lists... 
	prefdict = {}
	for i in range(prefmatrix.shape[0]):
		tmplist = list()
		for j in range(prefmatrix.shape[1]):
			tmplist.append(prefmatrix[i,j])
		prefdict[i+1] = tmplist
	allocation = {}
	number_of_players = prefmatrix.shape[0]
	t = 0
	while number_of_players>1:
		t = t+1
		tpref = np.zeros((prefmatrix.shape[0],prefmatrix.shape[0]))
		#print("First entry != 1",np.where(prefmatrix != -1))
		for i in prefdict.keys():
			firstpref = prefdict[i][0]
			tpref[i-1,firstpref-1] = 1 # Get the t-th column
		G = nx.from_numpy_matrix(tpref,create_using=nx.DiGraph())
		cycles = nx.simple_cycles(G)
		#print("cycles")
		cycles = list(cycles)
		for c in cycles: 
			for p in c:
				allocation[p+1] = prefdict[p+1][0]
		for c in cycles:
			for p in c:
				for l in prefdict.values():
					l.remove(p+1)
				number_of_players -= 1
		#print("Number of Players",number_of_players)
		#print("prefdict",prefdict)
	#print("t",t)
	return allocation	'''
		
def test_ttca():
	prefmatrix = np.matrix([[5,2,1,3,4],[1,2,3,4,5],[3,2,1,5,4],[4,3,2,5,1],[5,4,3,2,1]])
	allocation_2 = Ttca.ttca_ordered_set(prefmatrix)
	print("Allocation",allocation_2)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped	
	
def big_random_ttca():
	num_players = 10000
	prefmatrix = np.zeros((num_players,num_players))
	for i in range(num_players):
		prefmatrix[i] = np.random.choice(range(1,num_players+1),num_players,replace=False)
	prefmatrix = prefmatrix.astype(int)
	#print(prefmatrix)
	wrapped = wrapper(Ttca.ttca_ordered_set,prefmatrix)
	t = timeit.timeit(wrapped,number=1) # 0.937773827160493
	print(t)
	#print(allocation)
	
big_random_ttca()