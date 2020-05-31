'''every neural net and trainer need to have precisely the
same external shape, which is the number of inputs and the
number of outputs, and the domain of the inputs and outputs.
The domain is usually assumed to be 0.0 to 1.0 for each value. A neural net
also includes hidden layers, but the trainer evaluates each
neural net without observing anything about the hidden layers.
A trainer is an iterator which can give random (possibly from
a given list) training examples, which include an input point
and the expected (zero error) output point, in the form of a
tuple of two numpy column vectors, (input, ex_output)



Input and output values are passed between trainers and neural
nets as nnlists

'''

import numpy as np

def default_uniform_random_function():
    return np.random.uniform(low=0., high=1.)
def default_normal_random_function():
    return np.random.normal(loc=0., scale=1.)

class nnlist(list):

    def __init__(self, inval):
        if (type(inval)== np.ndarray):
            super().__init__(nnlist._fromvec_list(inval))
        else:
            super().__init__(inval)
        
    def vec(self):
        '''this method is nearly the whole reason this subclass of
list was created. It simply reforms this list into a usable numpy
column vector of the proper shape and dimension (2 dimensions, with
exactly 1 column)'''
        return np.array(self).reshape(len(self), 1)

    #static creation methods
    def _rand_nnlist(length, randfunc):
        '''returns an nnlist of this length, with each value
determined by a given random function. Use nnlist.uniform or
nnlist.normal for those distributions.'''
        return nnlist([randfunc() for x in range(length)])
    def uniform(length):
        return nnlist._rand_nnlist(length, default_uniform_random_function)
    def normal(length):
        return nnlist._rand_nnlist(length, default_normal_random_function)
    def _fromvec_list(array):
        '''create a regular python list from a numpy column vector. A warning
will be given if array is not a numpy column vector (two dimensional
vector, with n rows (first dimension) and 1 column (second dimension)
this is a helper function for the initializer'''
        if(array.ndim != 2):
            raise Exception("tried to create an nnlist from a numpy array without two dimensions")
        if(array.shape[1] != 1):
            raise Exception("tried to create an nnlist from a numpy array that wasn't a column (second dimension size wasn't 1)")
        return [x.item() for x in np.nditer(array)]






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
