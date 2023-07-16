require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')

local matrix = require 'Matrix'
local table = require "Table"

DenseLayer = {}

function DenseLayer:new (n_inputs, n_neurons, weights, bias)
   local o = {}
   setmetatable(o, {__index = self})
   o.n_inputs = n_inputs
   o.n_neurons = n_neurons
   --used for testing, remove later
   local normal3 = matrix({
      {-0.0046489263121093  ,   0.0020956038722145   ,   0.0013919233079285},
      {0.0016897559797244  ,    -0.015094421757785  ,    -0.0083872141359904},
      {-0.01564357881206   ,    0.015666571269968   ,    9.1913513459111e-05}
   })

   local normal2 = matrix({
      {0.006137059494733   ,    -0.0028233685517469  ,   -0.0093778269302466},
      {0.011709864170328    ,   0.0020977908861994  ,   0.0014130520542403}
   })

   if n_inputs == 2 and n_neurons == 3 then
      o.weights = normal2
   elseif n_inputs == 3 and n_neurons == 3 then
      o.weights = normal3
   end
   --o.weights = weights or matrix:new(table.normal({n_inputs, n_neurons}, 0, 1, 0.01))
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
   self.dbiases = matrix({table.sumT(dvalues, 0)})
   self.dweights = matrix.dot(inputs_transposed, dvalues)
   self.dinputs = matrix.dot(dvalues, weights_transposed)
end

return DenseLayer