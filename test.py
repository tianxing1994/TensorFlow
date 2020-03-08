import numpy as np

a = np.array(['jack', 'bryan', 'miyuki'], dtype=np.string_)
print(a)
print(a.dtype)

string = str(a[0], 'utf-8')
print(string)
print(type(string))