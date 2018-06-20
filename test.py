import pandas
import numpy as np
b = ['a','b','c','d','e']
a = pandas.Series(np.array([1,2,3,4,5]),index=b)
print(a['a'])