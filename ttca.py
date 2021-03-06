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
		
class TtcaBaseClass():
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
	def init_from_csv_file(self,csv_path):
		with open(csv_path,'rb') as f:
			reader = csv.reader(f)
			print(list(reader))
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
		
class OrderedDictTtca(TtcaBaseClass):
	prefdict = {}
	def init_data_structures(self):
		for i in range(1,self.total_number_of_players+1): # Foreach player
			tmpdict = ODict() # Create an OrderedDict
			tmpdict = ODict.fromkeys(list(self.prefmatrix[i-1]),None)
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

class Ttca(TtcaBaseClass):
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
	
