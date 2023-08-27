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
   o.rememberTrainableLayers = function (trainableLayers) return self:RememberTrainableLayers(o, trainableLayers) end
   return o
end

function BaseLoss:Calculate(self, output, y)
   output = output or self.output
   y = y or self.y

   local sample_losses = self.forward(output, y)
   --DONT FORGET TO RE ADD THIS!!!!
   return table.avg(sample_losses)--, self.regularization_loss()
end

function BaseLoss:RegularizationLoss(self)
   local reg_loss = 0

   for i = 1, #self.trainableLayers do
      local layer = self.trainableLayers[i]
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
   end

   return reg_loss
end

function BaseLoss:RememberTrainableLayers(self, trainableLayers)
   self.trainableLayers = trainableLayers
end

return BaseLoss


