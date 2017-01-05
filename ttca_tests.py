import unittest
import numpy as np
from ttcalib import Ttca,ODict,TtcaBaseClass
import timeit
import sys
import csv


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

class TestBaseClassMethods(unittest.TestCase): 
	def test_csv_reader(self):
		ttca = TtcaBaseClass(prefmatrix=None)
		prefmatrix = np.array([[1,5,2,1,3,4],[2,1,2,3,4,5],[3,3,2,1,5,4],[4,4,3,2,5,1],[5,5,4,3,2,1]])
		with open('tmp.csv','w',newline='') as csvfile: 
			csvwriter = csv.writer(csvfile,delimiter=';')
			for i in range(prefmatrix.shape[0]):
				csvwriter.writerow(list(prefmatrix[i]))
		ttca.init_prefmatrix_from_csv_file('tmp.csv',delimiter=';')
		np.testing.assert_array_equal(prefmatrix[:,range(1,6)],ttca.prefmatrix)
		
	
class TestTtcaMethods(unittest.TestCase):
	def test_ttca(self):
		print("Test Ttca")
		prefmatrix = np.array([[5,2,1,3,4],[1,2,3,4,5],[3,2,1,5,4],[4,3,2,5,1],[5,4,3,2,1]])
		ttca = Ttca(prefmatrix)
		allocation = ttca.calculate_allocations()
		
		self.assertEqual(allocation,{5:5,4:4,3:3,1:2,2:1})
	
	def test_big_random_ttca(self):
		num_players = 100
		prefmatrix = np.zeros((num_players,num_players))
		for i in range(num_players):
			tmprow = list(range(1,num_players+1))
			np.random.shuffle(tmprow)
			prefmatrix[i] = tmprow
		prefmatrix = prefmatrix.astype(int)
		print("Starting Benchmark...")
		ttcac = Ttca(prefmatrix,show_progress=True,progress_function=update_progress)
		wrapped = wrapper(ttcac.calculate_allocations)
		t = timeit.timeit(wrapped,number=1) # 0.937773827160493
		print("Needed",t,"seconds for a",num_players,"players instance")
		
class TestFirstDict(unittest.TestCase):
	def test_get_empty_item(self):
		emptydict = ODict()
		with self.assertRaises(StopIteration):
			emptydict.get_first()
	def test_get_first_item(self):
		testdict = ODict()
		testdict[0] = 999
		self.assertEqual(testdict.get_first(),0)
	def test_items_are_retrieved_in_order(self):
		testdict = ODict()
		testdict['first']= 'firstitem'
		testdict['second'] = 'seconditem'
		self.assertEqual(testdict.get_first(),'first')
	
if __name__ == '__main__':
	unittest.main()