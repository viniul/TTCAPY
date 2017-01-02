import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
from collections import OrderedDict
import timeit
import csv
import sys
import heapq


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


class ODict(OrderedDict):
	''' Inherit from OrderedDict, but add the function get_first '''
	def get_first(self):
		''' Return the first element of the OrderedDict'''
		return next(iter(self))
		
class Ttca():
	''' Take prefmatrix, output which player gets which house'''
	'''Prefmatrix: i-th row, j-th column: player i ranks player M[i,j] on the j-th place'''
	''' whereby M[i,j] \in {1,...,k} \forall i,j, i.e. Players are denoted with 1...keys'''
	''' Return: A dictionary, whereby dict[i] is the house that player i gets'''
	prefmatrix = None
	total_number_of_players = 0
	number_of_players = 0
	cycles = None
	allocation = {}
	def init_data_structures(self):
		''' This function is supposed to init the data structures (e.g. orderedDict, Matrix, etc.)'''
		raise Exception('Not implemented')
	def retrieve_edges(self):
		''' Retrieve the next edges_list for round t'''
		raise Exception('Not implemented')
	def process_cycles(self):
		''' Given the cycles of round t, process the resulsts, i.e. update data structures'''
		raise Exception('Not implemented')
	def __init__(self,prefmatrix,show_progress=True,progress_function=update_progress):
		self.prefmatrix = prefmatrix
		self.show_progress = show_progress
		self.progress_function = progress_function
	def calculate_allocations(self):
		''' Calculate the allocations with the help of the class functions'''
		self.total_number_of_players = self.number_of_players = self.prefmatrix.shape[0]
		self.init_data_structures()
		while self.number_of_players>=1:
			G = nx.DiGraph()
			# Create the graph: Each player "points" to the player he likes best out of the remaining players
			edges_list = self.retrieve_edges()
			G.add_edges_from(edges_list)
			# Find all the cycles in the graph, i.e. all the players that would be better off by trading
			cycles = nx.simple_cycles(G)
			self.cycles = list(cycles)
			self.process_cycles()
			if self.show_progress==True:
				self.progress_function(1-(float(self.number_of_players)/self.total_number_of_players))
		return self.allocation
		
class OrderedDictTtca(Ttca):
	prefdict = {}
	def init_data_structures(self):
		for i in range(1,self.total_number_of_players+1): # Foreach player
			tmpdict = ODict() # Create an OrderedDict
			tmpdict = ODict.fromkeys(list(self.prefmatrix[i-1]),None)
			#for j in range(number_of_players): #
			#	tmpdict[prefmatrix[i-1,j]] = None # Add the rank as the j-th keys, i.e. players are inserted in the dict
                # in the same order as player i ranks them
			self.prefdict[i] = tmpdict
			
	def retrieve_edges(self):
		return map(lambda i: (i,self.prefdict[i].get_first()),self.prefdict.keys())
	def process_cycles(self):
		for c in self.cycles:
				for p in c:
					self.allocation[p] = self.prefdict[p].get_first()
		for c in self.cycles:
				for p in c:
					for d in self.prefdict.values():
						del d[p]
					del self.prefdict[p]
					self.number_of_players -= 1

class ColumnWiseTtca(Ttca):
	''' Transform the prefmatrix, such that each row represents the preference list of one player '''
	''' And the entry of the j-th column of the row denotes the ranking (one,two,three..) of the j-th player's resource'''
	''' -1 means player deleted'''
	pref_list = None			
	def init_data_structures(self):
		d = self.total_number_of_players+1
		self.tmatrix = np.zeros((d-1,d))
		self.tmatrix = self.tmatrix.astype(int)
		for i in range(0,self.total_number_of_players):
			self.tmatrix[i,0] = i
			for j in range(0,self.total_number_of_players):
				self.tmatrix[i,self.prefmatrix[i,j]] = j
	def retrieve_edges(self):
		player_list = list(self.tmatrix[:,0])
		self.pref_list = np.argmin(self.tmatrix[:,1:],axis=0) # Get the first preference for each player
		edges_list = list(zip(player_list,self.pref_list))
		return edges_list
	def process_cycles(self):
		for c in self.cycles:
			for p in c:
				self.allocation[p+1] = self.pref_list[p]+1 # If p is in a cycle, then p got his best preference in round t 
				mask = (self.tmatrix[:,0] != p)
				self.tmatrix = self.tmatrix[mask]
				self.tmarix = np.delete(self.tmatrix,c,1)
				self.number_of_players -= len(c)
		#print(self.tmatrix)
	
'''	
class Ttca():
	def ttca_rows_as_player(prefmatrix,show_progress=True,progress_function=update_progress):
	
	def ttca_ordered_set(prefmatrix,show_progress=True,progress_function=update_progress):
		 Take prefmatrix, output which player gets which house
		Prefmatrix: i-th row, j-th column: player i ranks player M[i,j] on the j-th place
		 whereby M[i,j] \in {1,...,k} \forall i,j, i.e. Players are denoted with 1...keys
		 Return: A dictionary, whereby dict[i] is the house that player i gets
		prefdict = {} # Prefdict[i] = Preference list of player i as an OrderedDict
		number_of_players = prefmatrix.shape[0]
		for i in range(1,number_of_players+1): # Foreach player
			tmpdict = ODict() # Create an OrderedDict
			tmpdict = ODict.fromkeys(list(prefmatrix[i-1]),None)
			#for j in range(number_of_players): #
			#	tmpdict[prefmatrix[i-1,j]] = None # Add the rank as the j-th keys, i.e. players are inserted in the dict
                # in the same order as player i ranks them
			prefdict[i] = tmpdict
		allocation = {} # Allocation dict, player i gets house of allocation[i]
		total_number_of_players = prefmatrix.shape[0]
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

	min_player = 1 # Are players indexed by zero,one,etc.?
	def get_first_pref(row):
		 Upon receiving a row-preference vector, get the first element that is not 0
		if len(row.shape)!=1:
			raise ValueError('First argument must be 1d vector')
		isnanidx = np.argmax(row > -1)
		# Note that, np.argmax returns 0 if no value is found - we need to check for that
		if isnanidx == 0 and row[isnanidx]<=-1:
			raise ValueError('The given row contains no finite numbers')
		else:
			return row[isnanidx] 
	def ttca_vectorized(prefmatrix: np.ndarray,show_progress=True,progress_function=update_progress): 
		# Take prefmatrix, output which player gets which house
		#Prefmatrix: An ndarray where i-th row, j-th column: player i ranks player M[i,j] on the j-th place
		# whereby M[i,j] \in {1,...,k} \forall i,j, i.e. Players are denoted with 1...keys
		# Return: A dictionary, whereby dict[i] is the house that player i gets
		if type(prefmatrix)!=np.ndarray:
			raise TypeError('Preference matrix must be of type numpy.ndarray')
		if prefmatrix.shape[0]!=prefmatrix.shape[1]:
			raise ValueError('Prefmatrix must be quadratic')
		#TODO: Before more sanity checks, e.g. is the preference list of every player unique 
		print("Start")
		min_player = Ttca.min_player
		prefdict = {} # Prefdict[i] = Preference list of player i as an OrderedDict
		number_of_players = total_number_of_players =  prefmatrix.shape[0]
		remaining_player_set = set(range(min_player,total_number_of_players+min_player))
		allocation = {} # Allocation dict, player i gets house of allocation[i]
		while number_of_players>1:
			G = nx.DiGraph()
			# Create the graph: Each player "points" to the player he likes best out of the remaining players
			#first_pref_list = list(map(lambda i: (i,Ttca.get_first_pref(prefmatrix[i-min_player])),remaining_player_set))
			edges_list = list(map(lambda i: (i,Ttca.get_first_pref(prefmatrix[i-min_player])),remaining_player_set))
			G.add_edges_from(edges_list)
			# Find all the cycles in the graph, i.e. all the players that would be better off by trading
			cycles = nx.simple_cycles(G)
			cycles = list(cycles)
			#For each cycle, save the allocation and delete the player.
			#print("start looping")
			removed_players = set()
			for c in cycles:
				for p in c:
					##
					allocation[p] = Ttca.get_first_pref(prefmatrix[p-min_player])
				cset = set(c)
				remaining_player_set -= cset
				removed_players |= cset # Set union
			for p in removed_players:
				for r in remaining_player_set:
					prefmatrix[r-min_player][np.where(prefmatrix[r-min_player]==p)] = -1
				#prefmatrix[np.where(prefmatrix==p)] = -1
			number_of_players -= len(removed_players)
			#print("Done Looping")
			#print(number_of_players)
			if show_progress==True: 
				progress_function(1-(float(number_of_players)/total_number_of_players))
		return allocation


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
	return allocation	
'''	

def test_ttca():
	prefmatrix = np.array([[5,2,1,3,4],[1,2,3,4,5],[3,2,1,5,4],[4,3,2,5,1],[5,4,3,2,1]])
	ttca = ColumnWiseTtca(prefmatrix)
	allocation_2 = ttca.calculate_allocations()
	ttca = OrderedDictTtca(prefmatrix)
	allocation = ttca.calculate_allocations()
	print("allocation 1:",allocation)
	print("Allocation",allocation_2)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped	
	
def big_random_ttca():
	num_players = 10000
	prefmatrix = np.zeros((num_players,num_players))
	for i in range(num_players):
		tmprow = list(range(1,num_players+1))
		np.random.shuffle(tmprow)
		prefmatrix[i] = tmprow
	prefmatrix = prefmatrix.astype(int)
	#print(prefmatrix)
	print("starting")
	ttca = ColumnWiseTtca(prefmatrix)
	wrapped = wrapper(ttca.calculate_allocations)
	t = timeit.timeit(wrapped,number=1) # 0.937773827160493
	print("ColumnWise",t)
	ttca = OrderedDictTtca(prefmatrix)
	wrapped = wrapper(ttca.calculate_allocations)
	t = timeit.timeit(wrapped,number=1) # 0.937773827160493
	print("Vectorized",t)
	#print(allocation)

big_random_ttca()