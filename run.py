

import numpy as np
import matplotlib as mat
import myNeuralNet as mnn
import trainer as tr
import visualizer as vis



#sorta debugging, quick creation
exn = mnn.net([2,30,45,30,1])
exn2 = mnn.net([2,7,1])
extr = tr.example_c
extr2 = tr.example_t
t = np.array([[0.54],[0.23]])
extr2.fn(t)

vis.plotfun(extr2.fn, title="plot of the trainer function for 'limited trainer'")
vis.ploterror(extr2, exn2, title="plotting error of the example 2 neural net")

#vis.plotsgd(extr2, exn2)

##tm1 = np.random.randn(5,2)
##tm2 = np.random.randn(1,5)
##
##def tf(v):
##    value = np.dot(tm2, np.dot(tm1,v))
##    return value #already a numpy array


    
