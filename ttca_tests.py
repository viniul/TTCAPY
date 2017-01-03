import unittest
import numpy as np
from ttcalib import Ttca,OrderedDictTtca
import timeit
import sys


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped
	
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

class TestTtcaMethods(unittest.TestCase):
	def test_ttca(self):
		prefmatrix = np.array([[5,2,1,3,4],[1,2,3,4,5],[3,2,1,5,4],[4,3,2,5,1],[5,4,3,2,1]])
		#ttca = OrderedDictTtca(prefmatrix)
		#allocation = ttca.calculate_allocations()
		#self.assertEqual(allocation,{5:5,4:4,3:3,1:2,2:1})
	
	def test_big_random_ttca(self):
		num_players = 50
		prefmatrix = np.zeros((num_players,num_players))
		for i in range(num_players):
			tmprow = list(range(1,num_players+1))
			np.random.shuffle(tmprow)
			prefmatrix[i] = tmprow
		prefmatrix = prefmatrix.astype(int)
		#print(prefmatrix)
		print("starting")
		#ttcac = Ttca(prefmatrix)
		#wrapped = wrapper(ttcac.calculate_allocations)
		#t = timeit.timeit(wrapped,number=1) # 0.937773827160493
		#print("ColumnWise",t)
		ttcao = OrderedDictTtca(prefmatrix,show_progress=True,progress_function=update_progress)
		wrapped = wrapper(ttcao.calculate_allocations)
		t = timeit.timeit(wrapped,number=1) # 0.937773827160493
		print("OrderedDictTtca",t)
		#self.maxDiff = 6400
		#self.assertEqual(ttcac.calculate_allocations(),ttcao.calculate_allocations())
'''			
def test_ttca():
	
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
	num_players = 50
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
'''
if __name__ == '__main__':
	unittest.main()