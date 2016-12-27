import unittest
import numpy as np
from ttca import Ttca

class TestTtcaMethods(unittest.TestCase):

	def test_get_first_pref(self):
		''' Test on boundary values''' 
		x = np.zeros(1)
		x.fill(-1)
		x[0] = 1
		self.assertEqual(Ttca.get_first_pref(x), 1)
		x = np.zeros(500)
		x.fill(-1)
		x[499] = 50
		self.assertEqual(Ttca.get_first_pref(x), 50)
	def test_get_first_pref_for_exception(self):
		x = np.zeros(500)
		x.fill(-1)
		with self.assertRaises(ValueError) as context:
			Ttca.get_first_pref(x)

if __name__ == '__main__':
	unittest.main()