
BaseLoss = { output = {}, y = {} }

-- Derived class method new

function BaseLoss:new (o, output, y)
   local o = o or {}
   setmetatable(o, self)
   self.__index = self
   o.output = output
   o.y = y
   
   return o
end

function BaseLoss:Calculate(output, y)
   output = output or self.output
   y = y or self.y

   local sample_losses = self:Forward(output, y)
   -- add mean and return data loss
end

return BaseLoss


