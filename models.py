from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import numpy as np
import torch.nn as nn
from torch.nn.functional import log_softmax, sigmoid, tanh


class Flatten(nn.Module):
   def forward(self, input):
       return input.view(input.size(0), -1)

class HH2ComplexMLP(nn.Module):
  def __init__(self, input_size, device, n_outputs=1):
      super().__init__()

      self.flatten = Flatten()
      self.mlp_1 = ComplexLinear(input_size, input_size // 2)
      self.mlp_2 = ComplexLinear(input_size // 2, input_size // 4)
      self.mlp_3 = ComplexLinear(input_size // 4, input_size // 8)
      self.mlp_5 = ComplexLinear(input_size // 8, n_outputs)

  def forward(self, x):
      x = self.mlp_1(x)
      x = complex_relu(x)
      x = self.mlp_2(x)
      x = complex_relu(x)
      x = self.mlp_3(x)
      x = complex_relu(x)
      x = self.mlp_5(x)

      # Argument of x will map from -pi to pi and then convert it to have -180 to 180  
      output = (180/np.pi) * x.angle()
      
      return output
  