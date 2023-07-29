local table = require('Table')
local matrix = require('Matrix')
local table = require('Table')

BaseLoss = { output = {}, y = {} }

-- Derived class method new

function BaseLoss:new (o)
   local o = o or {}
   setmetatable(o, self)
   self.__index = self
   o.calculate = function (output, y) return self:Calculate(o, output, y) end
   o.regularization_loss = function (layer) return self:RegularizationLoss(o, layer) end
   return o
end

function BaseLoss:Calculate(self, output, y)
   output = output or self.output
   y = y or self.y

   local sample_losses = self.forward(output, y)

   return table.avg(sample_losses)
end

function BaseLoss:RegularizationLoss(self, layer)
   local reg_loss = 0

   if layer.weight_regularizer_l1 > 0 then
      reg_loss = reg_loss + layer.weight_regularizer_l1 * matrix.sum(matrix.abs(layer.weights))
   end

   if layer.weight_regularizer_l2 > 0 then
      reg_loss = reg_loss + layer.weight_regularizer_l2 * matrix.sum(matrix.square(layer.weights))
   end

   if layer.bias_regularizer_l1 > 0 then
      reg_loss = reg_loss + layer.bias_regularizer_l1 * matrix.sum(matrix.abs(layer.biases))
   end

   if layer.bias_regularizer_l2 > 0 then
      reg_loss = reg_loss + layer.bias_regularizer_l2 * matrix.sum(matrix.square(layer.biases))
   end
   
   return reg_loss
end

return BaseLoss


