import matplotlib.pyplot as plt
import numpy as np
import neuralnetutil as util



def test_function(point):
    '''the test function should return a two-dimensional vector
(with a single value)'''
    return np.array([[point[0]+point[1]]])
def runtest(title = "testing the visualizer"):
    plotfun(test_function)

def getcolor(v, maxv, minv, alpha=False):
    '''v is a scalar, not a vector or matrix. returns
the color corresponding to v, green if
close to minv, red if close to maxv. Always constrains the
value to lie between minv and maxv.'''
    if(maxv==minv):
        if(alpha): return (0.0, 1.0, 0.0, 1.0)
        else: return (0.0, 1.0, 0.0)

    normv = (v-minv)/(maxv-minv)
    if(normv>1.0): normv = 1.0
    if(normv<0.0): normv = 0.0

    if(alpha): return (normv, 1.0-normv, 0.0, 1.0)
    else: return (normv, 1.0-normv, 0.0)



def plotsgd(tr, net, eta=0.05, testpoints=5, batchsize=5, numupdates = 100):
    '''hoo boy this is a big un. updates the neural net using
the net's SGD, and plots the error after each batch update. after
numupdates it displays the graph.
testpoints is the number of points used to calculate the error after each update'''
    #sgd(training_data, mini_batch_size, eta, mini_batch_in_epoch)
    curve = [estimateerror(tr, net, testpoints)]
    for i in range(numupdates):
        net.my_SGD(tr, batchsize, eta, 1)
        curve.append(estimateerror(tr, net, testpoints))
    plt.plot(curve)
    plt.show()


def scalarerroratpoint(tr, net, point):
    return net.cost(point, tr.fn(point))

def estimateerror(tr, net, testpoints):
    totalerror = 0.0
    for i in range(testpoints):
        examplepoint = next(tr)
        totalerror += scalarerroratpoint(tr, net, examplepoint[1])
    return totalerror/float(testpoints)

        

def ploterror(tr,net, title="plotting error of net", minx=0.0, miny=0.0, maxx=1.0, maxy=1.0):
    '''this function takes a function approximation trainer,
which can take arbitrary points and return the expected value,
and a neural net which has possibly been trained to approximate
that trainer function. This function plots the error for the
net for a range of points in two dimensions between the min and max values
Using the error calculation in the net itself.'''
    def pointerror(v):
        err = net.errorpoint(v, tr.fn(v))
        return np.array([[err]])
    
    plotfun(pointerror, title, minx, miny, maxx, maxy)
    ##end function ploterror


    


def plotfun(fn, title="plot function", minx=0.0, miny=0.0, maxx=1.0, maxy=1.0):
    '''the default square is 0 to 1 along two
dimensions. fn is a function that takes a 2d
column vector with values between 0 and 1 and returns some
scalar value (in a two-dimensional numpy vector with a
single value, which is the default output of a neural
net.) '''

    numpoints = 60 #the default number of points along one dimension (total number of points is this value squared)

#okay so pointsList holds the points in [x,y] format, not numpy column vectors.
    pointsList = []

    for x in np.linspace(minx, maxx, numpoints, endpoint=True):
        for y in np.linspace(miny, maxy, numpoints, endpoint=True):
            point = (x,y)
            pointvector = np.array([[x],[y]])

            #debugging
##            print("how is pointvector considered two arguments?")
##            print(pointvector)

            fnoutput = fn(pointvector)
            scalarvalue = fnoutput[0,0]
            pointsList.append( (scalarvalue, point) )

    valueList = [p[0] for p in pointsList]
    
    maxv = max(valueList)
    minv = min(valueList)

    
    plt.title(title + "\nmax: " + str(maxv) + ", min: " + str(minv))
    
    for p in pointsList:
        plt.scatter(p[1][0], p[1][1], c=getcolor(p[0], maxv, minv), edgecolors="none")

    plt.show()
    ##end function plotfun

    
