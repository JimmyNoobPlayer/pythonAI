
import numpy as np
import neuralnetutil as util


def sigmoid(z):
    #temp = 1.0 + np.exp(-z)
    #return 1.0/temp
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) 

neg_slope = 0.1
def scalarleakyRelu(z):
    if(z>0): return float(z)
    else: return neg_slope*float(z)
LRelu = np.vectorize(scalarleakyRelu)

def scalarleakyRelu_prime(z):
    if(z>0): return 1.0
    else: return neg_slope
LRelu_prime = np.vectorize(scalarleakyRelu_prime)

#act = sigmoid
#act_prime = sigmoid_prime

act = LRelu
act_prime = LRelu_prime




##import matplotlib.pyplot as plt
##
##def linint(value, minimum, maximum):
##    '''value is clamped between 0 and 1. returns
##minimum if 0 (or less), maximum if 1 (or greater)'''
##    useValue = value
##    if(value<0.0): useValue = 0.0
##    if(value>1.0): useValue = 1.0
##    return minimum + value*(maximum-minimum)
##
##def normalize(value, minimum, maximum, clip=True):
##    '''returns a value as if minimum were zero and maximum were 1. (like a one-dim projection)
##does not clamp the value if the value happens to be larger than max or less than min'''
##    if(minimum == maximum): return 0.0
##    output = (value-minimum)/(maximum-minimum)
##    if(clip):
##        if(output>1.0): return 1.0
##        if(output<0.0): return 0.0
##    return output
##
##def getcolor(value, minimum, maximum, alpha=True):
##    '''minimum color is green (0,1,0) max color is red (1,0,0)'''
##    nv = normalize(value, minimum, maximum, clip=True)
##    ##r = linint(normalize(value,minimum,maximum),0.0,1.0)
##    ##g = linint(normalize(value,minimum,maximum),1.0,0.0)
##    if(alpha): return(nv, 1.0-nv, 0.0, 1.0)
##    else: return(nv, 1.0-nv, 0.0)
    
##
##def ploterror(inputvector, error, maxerror):
##    '''minerror is assumed to be zero. Plot red points at high error, green at low,
##the inputvector is a one-dimensional numpy vector'''
##    plt.scatter(inputvector[0], inputvector[1], c=getcolor(error, 0.0, maxerror), edgecolors="face")
##
##
##def circle(vector):
##    incirclevalue = 0.9
##    outcirclevalue = 0.05
##    radsq = 0.25
##    length = 0.0
##    for x in vector:
##        length += x*x
##    if (length <= radsq): return incirclevalue
##    else: return outcirclevalue
##        
##    
##
##class function_approximator_trainer:
##    def __init__(self, length, fn):
##        self.length = length
##        self.fn = fn
##    def __iter__(self):
##        return self
##    def __next__(self):
##        vector = np.random.uniform(0.0,1.0, [self.length, 1])
##        output = fn(vector)
##        return (vector, output)
##
##circletrainer = function_approximator_trainer(2, circle)
##
##
##
##class simple_training_data:
##    '''this iterator of training data expects the output to be identical to the input, which is a list (actually a numpy column vector) of random values between 0 and one.
##    uses numpy.random.uniform, which has the default low of 0.0 and high of 1.0
##    '''
##    def __init__(self, length):
##        self.length = length
##    def __iter__(self):
##        return self
##    def __next__(self):
##        vector = np.random.uniform(0.0, 1.0, [self.length, 1])
##        return (vector, vector)
##    
##
##class my_td:
##    '''this is a set of training examples that detect points that
##are within a distance from the point in the center of that latent
##space. Two dimensions, 0 to 1 for each dimension, the center is
##0.5,0.5 and it senses points within 0.4 euclidean distance (gives
##an output of 0.9 for points in the ball, 0.1 for points outside that ball.)
##'''
##    ##def __init__(self):
##    def __iter__(self):
##        return self
##    def __next__(self):
##        vector = np.random.uniform(0.0, 1.0, [2,1])
##        a = vector[0,0]-0.5
##        b = vector[1,0]-0.5
##        if(a*a + b*b < 0.16):
##            expected = 0.9
##        else:
##            expected = 0.1
##        return (vector, np.array([[expected]]))
##    
        

#the numpy method array() creates ndarrays, the default type of numpy's special math arrays.
#numpy.random.rand(dimesion1, dim2, dim3, dimx) returns an ndarray of random values, uniform between 0 and 1. perfect for neural nets.
'''
class net(object):

    #init to random biases and weights. the biases and weights are the key thing that will be changed when the net learns.
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y,1) for y in sizes[1:]] # a python list of numpy python arrays
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:1-], sizes[1:])]
a(L) = phi( W(L)*a(L-1) + b(L) )
layer L is n large,
layer L-1 is m large,
W(L) is n by m (n columns, m rows), so W(L)xy is the weight from the yth neuron in L-1 to the xth neuron in L (you might think this is backwards but it's correct)
outputs are column vectors, mx1 for L-1, nx1 for L.
'''
class net(object):
    def __init__(self, sizes):
        '''sizes is a list giving the number of neurons in every layer, includng inputs and outputs. The first value is the number of inputs.'''
        self.numLayers = len(sizes)
        assert self.numLayers >= 2, "a neural network requires at least two layers at the extreme minimum to use the backpropagation algorithms, this had only " + str(self.numLayers)
        
        # this counts the inputs as a layer, so the 'net with 3 inputs, a "hidden" layer of 4 neurons then
        #2 outputs will be size [3,4,2]. The output of every layer as well as the output of the entire net will
        #have the same activation function.
        self.sizes = sizes
        #list comprehensions
        #these are lists of matrices, the size of which include self.numLayers-1 elements.
        self.biases = [np.random.randn(n,1) for n in sizes[1:]]
        self.weights = [np.random.randn(n,m) for m,n in zip(sizes[:-1], sizes[1:])] #switching of n,m is intentional




#I think this needs to use (np.array(list)).reshape(n,1) if you intend to use regular python arrays as input.
    def feedforward_reshape(self, a):
        '''Return the output of the network if "a" is input. a can be a simple list, anything that can be reshaped into a column vector.'''
        #n.feedforward(zarr.reshape(-1,1))
        #assert(len(a) == self.inputSize())
        useArr = (np.array(a)).reshape(self.sizes[0],1)
        for b,w in zip(self.biases, self.weights):
            useArr = act(np.dot(w,useArr)+b)
        return useArr

    def feedforward(self, inputArray):
        '''Return the output of the network if "a" is input. a
must be a numpy column vector (array), not just a list.'''
        a = inputArray
        for b,w in zip(self.biases, self.weights):
            nexta = act(np.add(np.dot(w,a), b))
            a = nexta
        return a

    def errorpoint(self, a, expected):
        return self.cost(self.feedforward(a), expected)




    def test_error(self, training_data, num_tests, debug=False):
        '''returns the average error for this number of examples taken from the training_data iterator (scalr, float)'''
        s = 0.0
        for x in range(0,num_tests):
            training_point = next(training_data)
            if(debug):
                print("input: " + str(training_point[0]))
                print("expected output: " + str(training_point[1]))
            result = self.feedforward(training_point[0])
            if(debug):
                print("actual output: " +str(result))
                print( "total cost or error: " + str(self.cost(result, training_point[1])) )
            s += self.cost(result, training_point[1])
        return float(s)/float(num_tests)
        #end method test_error


    def plot_errors(self, training_data, num_tests, approx_max_error):
        sum = 0.0
        for x in range(0,num_tests):
            training_point = next(training_data)
            result = self.feedforward(training_point[0])
            error = self.cost(result,training_point[1])
            ##print(training_point)
            ploterror(training_point[0], error, approx_max_error*1.05)
        plt.show()
                    

          




    def my_SGD(self, training_data, mini_batch_size, eta, mini_batch_in_epoch):
        '''Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a generator of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory numbers. (integer mini_batch_size and mini_batch_in_epoch, float eta.)

        training_data is assumed to be randomly generated and
        non-terminating. If it does terminate, my_SGD could produce an error.
        The training_data iterator could loop over a precalculated set of data.'''

        for j in range(mini_batch_in_epoch):
            
            mini_batch = [next(training_data) for x in range(0,mini_batch_size)]
            #I think a simple list comprehension is the best way to do that. (take a
            #bunch of values from the iterator and stick them in a list)
            
            self.update_mini_batch(mini_batch, eta)
            #since I don't do error tracking and my training_data iterator is
            #simpler, my_SGD is comparatively short




        #"update mini batch" process from the online book:
        #1 input a set of training examples (a list, created from the training set iterator given to mySGD)
        #2 for each training example x:

            #(this section is the backprop algorithm)
            #2.1 Feed forward finding z_xL and a_xL (the results and
            #activations for every point in the neural network for
            #this training example x)
            #2.2 FInd the "node error" for the final layer
            #2.3 Find the node error for every node, using the backprop algorithm.
            #2.4 sum up the error for every point, store in nabla_delta
            
        #3 change the neural network to make the "error" smaller according to the algorithm.

            
    def update_mini_batch(self, mini_batch, eta):
        '''Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.

        The bulk of the work is done by the line,
        delta_nabla_b, delta_nabla_w = self.backprop(x,y)
        at least the bulk of the work that requires mathematical
        knowledge rather than just organization skills.
        '''

        #nabla_b is the gradient (of the Cost or Total Error) with respect to this group of b values.
        #(nabla_w is the same for weights)
        # in programming terms, nabla_b is a list that holds every change that shoulld be done to
        #the biases after the training due to this mini-batch is complete.
        #delta_nabla_b is hilariously the change in this value due to a single training example.
        #the copied code uses the term "nabla" which is the typographical upside down triangle.
        #this symbol is often read "del" but that term has too many other meanings in programming contexts.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #all math here
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]     




    #And this is the actual backpropagation method
    def backprop(self, x, y):
        '''Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights". (storing the values of
        the needed changes to the biases and weights, for every value in the 'net)
        '''



        #process from the online book:
        #1 input a set of training examples (a list, created from the training set iterator given to mySGD (done in update_mini_batch)
        #2 for each training example x:
            #2.1 Feed forward finding z_xL and a_xL (the results and
            #activations for every point in the neural network for
            #this training example x)
            #2.2 FInd the "node error" for the final layer
            #2.3 Find the node error for every node, using the backprop algorithm.
            #2.4 sum up the error for every point, store in nabla_delta
        #3 change the neural network to make the "error" smaller according to the algorithm. (done in update_mini_batch)

        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward section same as the previous feedforward function, but keep track of all z and a.
        #of course the x value given is the input to the neural net, considered the activation of
        #the zeroth layer (the size of sizes[0]), and these values don't have a corresponding z value array.
        # we should add a dummy zero nparray to make the sizes of the activations and zs lists the same.
        #lol, I wanted to use the variable name a and as for a single activation and the complete list, but
        #"as" is a reserved keyword. use a, as_list, z, zs_list instead.
        a = x.copy()
        as_list = [a] # list to store all the activations, layer by layer
        zs_list = [] # list to store all the z vectors, layer by layer. This has one fewer value than the a list.
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            zs_list.append(z)
            a = act(z)
            as_list.append(a)
        #end feedforward section

        #debugging
##        print("length of weights list: " +str(len(self.weights)))
##        print("length of zs_list and as_list: " + str(len(zs_list)) + ", " +str(len(as_list)))
            
        # backward pass (use the "fundamental backpropagation equations")

        #calculate the node error for the final layer (output layer):
        delta = self.cost_derivative(as_list[-1], y) * act_prime(zs_list[-1]) #hadamard, or element-wise multiplication
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, as_list[-2].transpose())

        #debugging
##        print(nabla_b[-1])
##        print("and the weight of the output layer...")
##        print(nabla_w[-1])
        
        # Note that the variable l (renamed to j) in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
#for the backpropagation, count from the end of the list. This allows the various lists to have different lengths. Specifically, the list of a vectors has a "zeroth" vector representing the inputs.


        #index j counts from 2 to numLayers-1, using negative indexes
        #index -j starts at -2 which is the second to last layer in every list (last layer corresponds to outputs, which has a different mathematical calculation)
        #index -j continues until self.numLayers

        #the self.biases and self.weights lists have numLayer-1 elements.
        #the list of activations will have the one more element, the extra corresponding to the inputs.
        #the list of z values will have numLayer-1 also.
        #The index of [-1] will correspond to the final layer of a and z, layer F
        #The index of [-1] will correspond to the weights and biases for layer F (they produce the output for this layer, the output of the entire neural net.)

        for j in range(2, self.numLayers-1): #in python 3.x, range acts as xrange in python 2.x (producing iterators instead of lists)
            #debugging
##            print("in the math function, j=" + str(j))
##            print("self.numLayers is " + str(self.numLayers))
##            print("len of self.weights: " + str(len(self.weights)))
            z = zs_list[-j]
            sp = act_prime(z)
            delta = np.dot(self.weights[-j+1].transpose(), delta) * sp #using previously calculated delta. s*t is the hadamard or elementwise multiplication.
            nabla_b[-j] = delta
            nabla_w[-j] = np.dot(delta, as_list[-j-1].transpose()) #calculating this value for layer 0 requires the activations of "layer -1" the inputs.
        return (nabla_b, nabla_w)




    def cost_derivative(self, output_activations, expected_values):
        '''Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        for the cost function C = 1/2 * (output - y)^2
        '''
        return (output_activations-expected_values) #it's that easy!

    def cost(self, output_activations, expected_values):
        '''returns a single scalar value of the entire cost for this training element, C = 1/2 * (output - y)^2
        '''
        output = 0.0
        for a,y in zip(output_activations.flatten(), expected_values.flatten()):
            ##print(str(a))
            ##print(str(y))
            ##a_scalar = a.asscalar()
            ##y_scalar = y.asscalar()
            output += (a-y)*(a-y)
        return output * 0.5

testnet = net([2,7,7,1])



#This is a major section of copypasta directly from the online book (SGD and update_mini_batch)
            
##    def SGD(self, training_data, epochs, mini_batch_size, eta,
##            test_data=None):
##        """Train the neural network using mini-batch stochastic
##        gradient descent.  The "training_data" is a list of tuples
##        "(x, y)" representing the training inputs and the desired
##        outputs.  The other non-optional parameters are
##        self-explanatory.  If "test_data" is provided then the
##        network will be evaluated against the test data after each
##        epoch, and partial progress printed out.  This is useful for
##        tracking progress, but slows things down substantially."""
##        if test_data: n_test = len(test_data)
##        n = len(training_data)
##        for j in xrange(epochs): #btw, python version 3.x doesn't use xrange, the range function now uses this functionality (iterator rather than list)
##            random.shuffle(training_data)
##            mini_batches = [
##                training_data[k:k+mini_batch_size]
##                for k in xrange(0, n, mini_batch_size)]
##            for mini_batch in mini_batches:
##                self.update_mini_batch(mini_batch, eta)
##            if test_data:
##                print "Epoch {0}: {1} / {2}".format(
##                    j, self.evaluate(test_data), n_test)
##            else:
##                print "Epoch {0} complete".format(j)
##
##    def update_mini_batch(self, mini_batch, eta):
##        """Update the network's weights and biases by applying
##        gradient descent using backpropagation to a single mini batch.
##        The "mini_batch" is a list of tuples "(x, y)", and "eta"
##        is the learning rate."""
##        nabla_b = [np.zeros(b.shape) for b in self.biases]
##        nabla_w = [np.zeros(w.shape) for w in self.weights]
##        for x, y in mini_batch:
##            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
##            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
##            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
##        self.weights = [w-(eta/len(mini_batch))*nw 
##                        for w, nw in zip(self.weights, nabla_w)]
##        self.biases = [b-(eta/len(mini_batch))*nb 
##                       for b, nb in zip(self.biases, nabla_b)]     
##    



#And this is the actual backpropagation method
## def backprop(self, x, y):
##        """Return a tuple "(nabla_b, nabla_w)" representing the
##        gradient for the cost function C_x.  "nabla_b" and
##        "nabla_w" are layer-by-layer lists of numpy arrays, similar
##        to "self.biases" and "self.weights"."""
##        nabla_b = [np.zeros(b.shape) for b in self.biases]
##        nabla_w = [np.zeros(w.shape) for w in self.weights]
##        # feedforward
##        activation = x
##        activations = [x] # list to store all the activations, layer by layer
##        zs = [] # list to store all the z vectors, layer by layer
##        for b, w in zip(self.biases, self.weights):
##            z = np.dot(w, activation)+b
##            zs.append(z)
##            activation = sigmoid(z)
##            activations.append(activation)
##        # backward pass
##        delta = self.cost_derivative(activations[-1], y) * \
##            sigmoid_prime(zs[-1])
##        nabla_b[-1] = delta
##        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
##        # Note that the variable l in the loop below is used a little
##        # differently to the notation in Chapter 2 of the book.  Here,
##        # l = 1 means the last layer of neurons, l = 2 is the
##        # second-last layer, and so on.  It's a renumbering of the
##        # scheme in the book, used here to take advantage of the fact
##        # that Python can use negative indices in lists.
##        for l in xrange(2, self.num_layers):
##            z = zs[-l]
##            sp = sigmoid_prime(z)
##            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
##            nabla_b[-l] = delta
##            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
##        return (nabla_b, nabla_w)
##
##...
##
##    def cost_derivative(self, output_activations, y):
##        """Return the vector of partial derivatives \partial C_x /
##        \partial a for the output activations."""
##        return (output_activations-y) 
##
##def sigmoid(z):
##    """The sigmoid function."""
##    return 1.0/(1.0+np.exp(-z))
##
##def sigmoid_prime(z):
##    """Derivative of the sigmoid function."""
##    return sigmoid(z)*(1-sigmoid(z))                       



        
