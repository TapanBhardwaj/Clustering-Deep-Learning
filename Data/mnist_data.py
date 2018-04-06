from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import gzip
import _pickle as cPickle

mnist = fetch_mldata('MNIST original')
print(mnist.data.shape)
print(mnist.target)
xtrain,xtest,ytrain,ytest=train_test_split(mnist.data,mnist.target,shuffle=True,test_size=0.1)


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(xtest.shape)
xfinal=np.concatenate((xtrain,xtest))
yfinal=np.concatenate((ytrain,ytest))
print(xfinal.shape)
print(yfinal.shape)

with gzip.open('mnist_dcn.pkl.gz','wb') as f:
    cPickle.dump([xfinal,yfinal],f)