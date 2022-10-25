import numpy as np
import matplotlib.pyplot as plt

#define the activation functions
def relu(z):
    return np.maximum(0,z)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def softplus(x):
    return np.log(1 + np.exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def plot_function(f,ax):
    x = np.arange(-5, 5 + 0.1, 0.1)
    ax.plot(x, f(x))
    title = f.__name__
    ax.set_title('%s' % title)
    
#plot the activation functions
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
plot_function(relu,axes[0,0])
plot_function(sigmoid,axes[0,1])
plot_function(softplus,axes[1,0])
plot_function(softmax,axes[1,1])
plt.tight_layout()
plt.show()
