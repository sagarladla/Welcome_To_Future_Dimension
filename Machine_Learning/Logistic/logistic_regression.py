import numpy as np

def hypothesis(thetas,X):
    z=-np.dot(X,thetas)
    denominator=1+np.exp(z)
    return (1/denominator)

def cost(thetas,X,y):
    M=X.shape[0]
    c = -y*np.log(hypothesis(thetas,X)) - (1-y)*np.log(1-hypothesis(thetas,X))
    c=c/M
    return c

def main():
    X=np.matrix([[1,24],[1,25],[1,22],[1,10],[1,15],[1,13]])
    y=np.matrix([[0],[0],[0],[1],[1],[1]])
     
    print(X)
    print(y)

main()
