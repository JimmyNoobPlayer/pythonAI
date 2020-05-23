'''this python file is dedicated to a "trainer", a
way to train a neural net to approximate a regular
ol' python function (which could also be part of a
multi-part training regimen, or a wrapper for the
usual example-based training used in most simple
online neural net examples.

The most important quality of the function is the
number of values in the vectors for input and output,
these numbers must match between the neural net being
trained, the trainer, and the function to be approximated.

The trainer will usually need a "translator" to translate inputs into the numpy column vector input, and translate the output of the net into 

The expected format of the training example points is
(vect, output) where vect is a numpy column vector
representing the input point, and output a numpy column
vector representing the output of the neural net.
'''
import numpy as np
import neuralnetutil as util
import random




    

class function_approx_trainer:
    def __init__(self, length, fn):
        '''length is the length of the input vector, fn
is the function the net will be trained to approximate.
Besides being iterators, the trainer should be able to
take valid input vectors and show the expected value of
the output vector.'''
        self.length = length
        self.fn = fn
    def __iter__(self):
        return self
    def __next__(self):
        v = util.default_get_input_point(self.length)
        return (v, self.fn(v))

##    def validate_input(self, v):
        

##def pyramid(v):
##    x = v[0,0]
##    y = v[0,1]
##
##    if(x<0.5):
##        if(y<0.5)
        


class limited_trainer:
    '''this function is only defined on five
points. two input values, the output is 0.1 on the
corners (0,0  0,1  1,1  1,0) and 0.9 on the center 0.5,0.5. the
output for any other point is not defined. when the trainer
selects a random point, it will only select one of these.

This is meant to be a trainer that is very simple to observe
and can be approximated by a neural net exactly.'''


    def __init__(self):
        self.fn = wedge #no variation in trainers, this could possibly be a static class, but I don't want to change anything if I can help it.
    def __iter__(self):
        return self
    def __next__(self):
        '''these dictionaries define the function'''
        possible_points = "abcde"
        points_selector = {"a":(0.0,0.0), "b":(0.0,1.0), "c":(1.0,0.0), "d":(1.0,1.0), "e":(0.5,0.5)}
        value_selector = {"a":0.1, "b":0.1, "c":0.1, "d":0.1, "e":0.9}

        r = random.choice(possible_points)
        rp = points_selector[r]

        #debugging
##        print("the point given in the trainer")
##        print(rp)
        
        rv = value_selector[r]
        point = np.array([[rp[0]],[rp[1]]])
        value = np.array([[rv]])

        return (point,value)
    
def wedge(v):
    '''takes a 2d column vector, returns a continuous function that matches the trainer at the 5 relevant points.'''
    #the return value is only a function of the first value in v
    inputval = v[0,0]
    if(inputval>0.5): value = 1.7-1.6*inputval
    else: value = 0.1+1.6*inputval
    return np.array([[value]])
    
    

def example_circlefunction(inv):
    incirclevalue = 0.95
    outcirclevalue = 0.05
    radsq = 0.25

    lsq = 0.0
    for x in inv:
        lsq+=(x-0.5)*(x-0.5)
    if(lsq<=radsq): return np.array([[incirclevalue]])
    else: return np.array([[outcirclevalue]])

def example_identityfunction(inv):
    return inv

def example_maxfunction(inv):
    '''should take a numpy column
vector (2 dimensional matrix) as input and return a
column vector as well.'''
    return np.array([[np.amax(inv)]])

example_c = function_approx_trainer(2, example_circlefunction)
example_m = function_approx_trainer(2, example_maxfunction)
example_t = limited_trainer()
        
