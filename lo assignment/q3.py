from q1 import *
import numpy as np

c=np.array([1,1,1,1,0,0,0,0,0,0,0,0])
b=np.array([1,1,1,1,1,1,1,1])
A=np.array([[1,1,0,0,0,0,0,0,-1,0,0,0],
            [0,1,1,0,0,0,0,0,0,-1,0,0],
            [0,1,0,1,0,0,0,0,0,0,-1,0],
            [0,0,1,1,0,0,0,0,0,0,0,-1],
            [1,0,0,0,1,0,0,0,0,0,0,0],
            [0,1,0,0,0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,1,0,0,0,0]])

x=revised_simplex(c,A,b)
if type(x)==str:
    print("The problem is unbounded")
else:
    print("The Optimal Solution is: ",np.dot(c,x))