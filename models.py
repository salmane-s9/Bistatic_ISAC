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
  
class HH2ComplexMLPClassifier(nn.Module):
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
        output = x.abs()

        return output
    
class HH2ComplexCONV_1(nn.Module):
  def __init__(self, input_size, device, n_outputs=1):
      super().__init__()

      self.flatten = Flatten()

      # Create Conv stack for each band covariance matrix
      self.conv_1 = ComplexConv2d(1, 1, kernel_size=(3,3), stride=1, padding=2)
      self.conv_2 = ComplexConv2d(1, 1, kernel_size=(3,3), stride=1, padding=2)
      self.bn_1  = ComplexBatchNorm2d(1)
      self.bn_2  = ComplexBatchNorm2d(1)
      self.mlp_1 = ComplexLinear(input_size, input_size // 2)
      self.mlp_2 = ComplexLinear(input_size // 2, input_size // 4)
      self.mlp_3 = ComplexLinear(input_size // 4, input_size // 8)
      self.mlp_4 = ComplexLinear(input_size // 8, n_outputs)

  def forward(self, x):
      
      x = self.conv_1(x)
      x = self.bn_1(x)
      x = complex_relu(x)      
      
      x = self.conv_2(x)
      x = self.bn_2(x)
      x = complex_relu(x)      
      x = x.view(x.size(0), -1)

      x = self.mlp_1(x)
      x = complex_relu(x)
      x = self.mlp_2(x)
      x = complex_relu(x)
      x = self.mlp_3(x)
      x = complex_relu(x)
      x = self.mlp_4(x)

      # Argument of x will map from -pi to pi and then convert it to have -180 to 180  
      output = (180/np.pi) * x.angle()
      
      return output
