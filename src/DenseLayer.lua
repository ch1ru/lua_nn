
require('Helper')
require('ActivationFunctions.Relu')
require('Tensor')

DenseLayer = {}

-- Derived class method new

function DenseLayer:new (o, n_inputs, n_neurons, weights, bias)
   o = o or {}
   setmetatable(o, self)
   self.__index = self
   self.n_inputs = n_inputs
   self.n_neurons = n_neurons
   self.weights = weights or Tensor:new(nil, table.normal({n_inputs, n_neurons}))
   self.bias = bias or Tensor:Zeros({1, n_neurons})
   
   return o
end

function DenseLayer:Forward(inputs)
   --apply relu activation
   local inputs = ReLU(inputs)
   self.output = (inputs * self.weights) + self.bias
end


local test = Tensor:new(nil, {{-2, -1, 0}})
local max = Tensor:Max(test)
print(max)
print(test)


