MSE = { y_pred = {}, y_true = {} }

require('BaseLoss')
require('Table')

local matrix = require('Matrix')

--Super method for calling parent constructor
MSE.Super = function (self) return BaseLoss:new(self) end

-- Derived class method crossentropyloss
function MSE:new()
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.forward = function (y_pred, y_true) return self:Forward(o, y_pred, y_true) end
    o.backward = function (dvalues, y_true) return self:Backward(o, dvalues, y_true) end
    return self.Super(o)
end

function MSE:Forward(self, y_pred, y_true)
    local sampleLosses = (matrix({y_true}) - matrix(y_pred))
    --FIXME!
    return sampleLosses
end

function MSE:Backward(self, dvalues, y_true)
    local samples = dvalues:columns() * dvalues:rows()
    local outputs = dvalues[0]:columns()
    self.dinputs = -2 * (y_true - dvalues) / outputs
    self.dinputs = self.dinputs / samples


end

return MSE