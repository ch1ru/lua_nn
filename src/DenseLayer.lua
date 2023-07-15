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
   --functions
   o.forward = function (inputs) return self:Forward(o, inputs) end
   o.backward = function (dvalues) return self:Backward(o, dvalues) end
   return o
end

function DenseLayer:Forward(self, inputs)
   self.inputs = inputs
   self.output = inputs * self.weights + self.bias
   return self.output
end

function DenseLayer:Backward(self, dvalues)
   local inputs_transposed = matrix.transpose(self.inputs)
   local weights_transposed = matrix.transpose(self.weights)
   self.dweights = matrix.dot(dvalues, inputs_transposed)
   self.dbiases = table.sumT(dvalues)
   self.dinputs = matrix.dot(dvalues, weights_transposed)
end