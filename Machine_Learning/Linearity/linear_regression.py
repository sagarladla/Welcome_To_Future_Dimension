import numpy as np
from matplotlib import pyplot as plt

def hypothesis(thetas,X):
    y = np.dot(X,thetas)
    return y

def loss(thetas,X,y):
    return hypothesis(thetas,X)-y

def cost(thetas,X,y):
    M=X.shape[0]
    temp=loss(thetas,X,y)
    sq=np.square(temp)
    summation = np.sum(sq)
    result = summation/(2*M)
    return result

def compute_partial_derivative(thetas,X,y):
    partial_thetas=(np.dot(X.T,loss(thetas,X,y)))
    return partial_thetas

def gradient_descent(X,y,thetas,epochs=3000,alpha=0.0005):
    M=X.shape[0]
    losses=[]
    iteration=[]
    for convergence in range(epochs):
        # print(convergence)
        losses.append(cost(thetas,X,y))
        iteration.append(convergence)
        gradient=compute_partial_derivative(thetas,X,y)/M
        thetas=thetas-alpha*gradient
    return thetas, iteration, losses

def read_file(filename='./ex1.txt'):
    matrix=[]
    y=[]
    with open(filename,'r') as file:
        for row in file:
            nrow=[]
            for r in row.split(','):
                nrow.append(float(r.strip()))
            matrix.append(nrow[:-1])
            y.append([nrow[-1]])
    return matrix,y

def standardize_data(X):
    std_data=X
    mean=np.mean(X,axis=0,dtype=np.longdouble)
    sub=np.subtract(X,mean,dtype=np.longdouble)
    sq=np.square(sub,dtype=np.longdouble)
    summation=np.sum(sq,axis=0,dtype=np.longdouble)
    summation=np.divide(summation,(X.shape[0]-1))
    std_deviation=np.sqrt(summation)
    std_data=np.divide(np.subtract(std_data,mean),std_deviation)
    return std_data

def main():
    X=np.matrix(read_file()[0],dtype=np.longdouble)
    X=standardize_data(X)
    one=np.matrix(np.ones(X.shape[0])).T
    X=np.append(one,X,axis=1)
    y=np.matrix(read_file()[1],dtype=np.longdouble)
    thetas=np.matrix(np.zeros(X.shape[1]),dtype=np.longdouble).T
    new_thetas,iteration,losses=gradient_descent(X,y,thetas)
    i=np.matrix([[1,5.5277]],dtype=np.longdouble)
    print(hypothesis(new_thetas,i))
    plt.plot(iteration,losses)
    plt.show()
if __name__ == "__main__":
    main()
