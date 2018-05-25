import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

s = {1,2,3,4,5}
print(1 in s)
s.add(1)

print(s)
print(6 in s)

def r():
    if 1==1:
        return True,True,True,True
    else:
        return 1,2,3,4

a,b,c,d = r()
print(a)

p = True

if not p:
    print('aaaa')

s = {1,2,3,4}
s.remove(1)
print(list(s))

print('qwe' == 'qw1')