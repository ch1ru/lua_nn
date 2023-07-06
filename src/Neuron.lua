require('BasicTensor.lua')

Neuron = {}

-- Derived class method new

function Neuron:new (o, input_features, bias)
   o = o or {}
   setmetatable(o, self)
   self.__index = self
   self.length = length or 0
   self.breadth = breadth or 0
   self.area = length*breadth;
   return o
end
