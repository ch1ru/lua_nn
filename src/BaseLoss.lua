
BaseLoss = { output = {}, y = {} }

-- Derived class method new

function BaseLoss:new (o, output, y)
   local o = o or {}
   setmetatable(o, self)
   self.__index = self
   o.output = output
   o.y = y

   o.calculate = function (output, y) return self:Calculate(o, output, y) end
   
   return o
end

function BaseLoss:Calculate(self, output, y)
   output = output or self.output
   y = y or self.y

   local sample_losses = self.forward(output, y)
  
   -- add mean and return data loss
   local dataLoss = table.avg(sample_losses)

   return dataLoss
end

return BaseLoss


