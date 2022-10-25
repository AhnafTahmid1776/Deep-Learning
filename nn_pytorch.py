#import the pytorch libraries
import torch
import torch.nn as nn

#layer 1: 3 inputs and 2 outputs with weights w and no bias
L_1 = nn.Linear(in_features=3, out_features=2, bias=False)
L_1.weight.data = torch.from_numpy(np.array([[3,-0.1,-1],[-1,0.50,2]],dtype=np.float64)).float()

#layer 2: 2 inputs and 1 output with weights H and no bias
L_2 = nn.Linear(in_features=2, out_features=1, bias=False)
L_2.weight.data = torch.from_numpy(np.array([-0.2,0.4],dtype=np.float64)).float()

#construct the complete network by stacking up layers, as well as the activation functions in between
model = nn.Sequential(L_1,
                      nn.ReLU(),
                      L_2,
                      nn.Sigmoid())

#define the input x: store the input array [2,-3,1] as a tensor
x = torch.from_numpy(np.array([2,-3,1], dtype=np.float64)).float()

#call the forward method to compute the output y
y_hat = model.forward(x)
print(y_hat)
