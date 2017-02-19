import networkx as nx
import numpy as np 
from collections import OrderedDict
import timeit
import csv
import sys


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
	def init_data_structures(self):
		''' This function is supposed to init the data structures (e.g. orderedDict, Matrix, etc.)'''
		raise Exception('Not implemented')
	def retrieve_edges(self):
		''' Retrieve the next edges_list for round t'''
		raise Exception('Not implemented')
	def process_cycles(self):
		''' Given the cycles of round t, process the resulsts, i.e. update data structures'''
		raise Exception('Not implemented')
	def __init__(self,prefmatrix,show_progress=False,progress_function=None):
		self.prefmatrix = prefmatrix
		self.show_progress = show_progress
		self.progress_function = progress_function
		self.allocation = {}
	def init_prefmatrix_from_csv_file(self,csv_path,delimiter=';'):
		''' Extract a prefmatrix from csv file'''
		''' The csv file should have the following format: One row for each player and '''
		''' every row should the name of the player's object in the first column followed by the preference ordering of this player, delimited by ; '''
		'''Returns: prefmatrix '''
		
		preflists = list()
		with open(csv_path, newline='') as csvfile:
			prefreader = csv.reader(csvfile,delimiter=delimiter)
			row_shape = None
			for row in prefreader:
				if row_shape is None: 
					row_shape = len(row)
				elif len(row)!=row_shape: 
						raise ValueError('Each preferencelist must be of the same length')
				preflists.append(row)
		if row_shape!=len(preflists)+1: 
			raise ValueError('The csv file does not contain the same amount of players as the preference rankings')
		# Preflists now contains one list for each player, with the desired objects in the correct order 
		self.obj_player_dict = {} # The dict that assigns each object to one player
		self.player_obj_dict = {} # The dict that assigns each player to one object
		for i in range(len(preflists)):
			self.obj_player_dict[preflists[i][0]] = i+1
			self.player_obj_dict[i+1] = preflists[i][0]
		prefmatrix = np.eye(len(preflists),dtype=int)		
		for i in range(0,len(preflists)): 
			for j in range(1,row_shape):
				prefmatrix[i,j-1] = self.obj_player_dict[preflists[i][j]]#preflists[i][j]
		self.prefmatrix = prefmatrix
		return self.prefmatrix
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
			# Once the cycles are found, process them, i.e. reallocate and remove players that were in their cycles. 
			self.process_cycles()
			if self.show_progress==True:
				self.progress_function(1-(float(self.number_of_players)/self.total_number_of_players))	
		return self.allocation
	def allocation_to_object_allocation(self):
		# Create a new allocation dict: 
		self.obj_allocation = {}
		for k in self.allocation.keys():
			self.obj_allocation[self.player_obj_dict[k]] = self.player_obj_dict[self.allocation[k]]
		return self.obj_allocation

class Ttca(TtcaBaseClass):
	prefdict = {}
	def __init__(self,prefmatrix,show_progress=False,progress_function=None):
		super(Ttca,self).__init__(prefmatrix,show_progress,progress_function)
	def init_data_structures(self):
		# Foreach player, create an OrderedDict, which contains his preferences in order.
		for i in range(1,self.total_number_of_players+1): 
			tmpdict = ODict() 
			tmpdict = ODict.fromkeys(list(self.prefmatrix[i-1]),None)
			self.prefdict[i] = tmpdict 
	def retrieve_edges(self):
		# The edges are created by getting the first preference u for each player 
		# And adding the edge 
		self.first_pref = {}
		edges_list = list(map(lambda i: (i,self.prefdict[i].get_first()),self.prefdict.keys()))
		self.first_pref = dict(edges_list) # Transform the list into a dict
		return map(lambda i: (i,self.prefdict[i].get_first()),self.prefdict.keys())
	def process_cycles(self):
		remove_keys = list()
		for c in self.cycles:
			for p in c:
				self.allocation[p] = self.first_pref[p]
				del self.prefdict[p]
				remove_keys.append(p)
		for p in remove_keys:
				for d in self.prefdict.values():
					del d[p]
		self.number_of_players -= len(remove_keys)