'''every neural net and trainer need to have precisely the
same external shape, which is the number of inputs and the
number of outputs, and the domain of the inputs and outputs.
The domain is usually assumed to be 0.0 to 1.0. A neural net
also includes hidden layers, but the trainer evaluates each
neural net without observing anything about the hidden layers.
A trainer is an iterator which can give random (possibly from
a given list) training examples, which include an input point
and the expected (zero error) output point, in the form of a
tuple of two numpy column vectors, (input, ex_output)


'''

import numpy as np

def default_random_function():
    return np.random.uniform(low=0., high=1.)

class nnlist(list):
        
    def vec(self):
        return np.array(self).reshape(len(self), 1)


def rand_nnlist(length):
    return nnlist([default_random_function() for x in range(length)])
        





def reshape(listvector):
    '''reshapes this python list into a numpy column vector'''
    return np.array(listvector).reshape(len(listvector), 1)

'''a domain class will be expected use floats and give the
dimensionality of the vector, and also be able to give a
random point from the domain. The default domain includes
every value from 0.0 to 1.0 inclusive for every dimension.
I don't forsee much need strictly defining other domains.'''
class nndomain:
    def __init__(self, dimension):
        self.d = dimension
    def rp():
        '''get a random point from this domain'''
        return np.random.uniform(0.0, 1.0, [self.d, 1])


def default_get_input_point(length):
    '''every trainer iterator needs to be able to
generate training points, which include an input vector
and the expected output. Sometimes the inputs should be
restricted in some way, but the default is to include
every value from 0 to 1 uniformly for every dimension of
the input vector.'''
    return np.random.uniform(0.0, 1.0, [length, 1])
