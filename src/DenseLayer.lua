require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')

local matrix = require 'Matrix'
local table = require "Table"

DenseLayer = {}

function DenseLayer:new (n_inputs, n_neurons, weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2)
   local o = {}
   setmetatable(o, {__index = self})
   o.n_inputs = n_inputs
   o.n_neurons = n_neurons
   o.weights = matrix:new(table.normal({n_inputs, n_neurons}, 0, 1, 0.01))
   o.biases = matrix:new({table.zeros(n_neurons)})
   o.weight_regularizer_l1 = weight_regularizer_l1 or 0.0
   o.weight_regularizer_l2 = weight_regularizer_l2 or 0.0
   o.bias_regularizer_l1 = bias_regularizer_l1 or 0.0
   o.bias_regularizer_l2 = bias_regularizer_l2 or 0.0
   --functions
   o.forward = function (inputs) return self:Forward(o, inputs) end
   o.backward = function (dvalues) return self:Backward(o, dvalues) end
   return o
end

function DenseLayer:Forward(self, inputs)
   self.inputs = inputs
   self.output = inputs * self.weights + self.biases
   return self.output
end

function DenseLayer:Backward(self, dvalues)
   
   local inputs_transposed = matrix.transpose(self.inputs)
   local weights_transposed = matrix.transpose(self.weights)
   self.dbiases = matrix({table.sumT(dvalues, 0)})
   self.dweights = matrix.dot(inputs_transposed, dvalues)

   if self.weight_regularizer_l1 > 0 then
      local dL1 = matrix:new({table.ones(self.weights:size())})
      for i = 1, self.weights:rows() do
         for j = 1, self.weights:columns() do
           if self.weights[i][j] < 0 then
             matrix.setelement(dL1, i, j, -1)
           end
         end
      end
      self.dweights = self.dweights + matrix.mulnum(dL1, self.weight_regularizer_l1)
   end

   if self.weight_regularizer_l2 > 0 then
      self.dweights = self.dweights + matrix.mulnum(self.weights, 2 * self.weight_regularizer_l2)
   end

   if self.bias_regularizer_l1 > 0 then
      local dL1 = matrix:new({table.ones(self.biases:size())})
      for i = 1, self.biases:rows() do
         for j = 1, self.biases:columns() do
           if self.biases[i][j] < 0 then
             matrix.setelement(dL1, i, j, -1)
           end
         end
      end
      self.dbiases = self.dbiases + matrix.mulnum(dL1, self.bias_regularizer_l1)
   end

   if self.bias_regularizer_l2 > 0 then
      self.dbiases = self.dbiases + matrix.mulnum(self.biases, 2 * self.bias_regularizer_l2)
   end

   self.dinputs = matrix.dot(dvalues, weights_transposed)
end

return DenseLayer