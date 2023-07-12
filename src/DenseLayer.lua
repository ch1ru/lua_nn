require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')

local matrix = require 'Matrix'
local table = require "Table"

DenseLayer = {n_inputs = 0, n_neurons = 0, weights = matrix:new({}), bias = matrix:new({}) }

-- Derived class method new

function DenseLayer:new (o, n_inputs, n_neurons, weights, bias)
   local o = o or {}
   setmetatable(o, self)
   self.__index = self
   o.n_inputs = n_inputs
   o.n_neurons = n_neurons
   o.weights = weights or matrix:new(table.normal({n_inputs, n_neurons}, 0, 1, 0.01))
   o.bias = bias or matrix:new({table.zeros(n_neurons)})
   return o
end

function DenseLayer:Forward(inputs)
   --apply relu activation
   self.output = inputs * self.weights + self.bias
   return self.output
end