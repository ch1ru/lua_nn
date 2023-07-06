require('Tensor.lua')

NeuronLayer = {}

-- Derived class method new

function NeuronLayer:new (o, input_features, num_output, weights, bias)
   o = o or {}
   setmetatable(o, self)
   self.__index = self
   self.input_features = input_features
   self.num_output = num_output
   self.weights = weights
   self.bias = bias

   --calculate output
   self.output = (input_features * weights) + bias
   return o
end
