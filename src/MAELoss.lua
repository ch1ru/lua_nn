MAE = { y_pred = {}, y_true = {} }

require('BaseLoss')
require('Table')

local matrix = require('Matrix')

-- Derived class method crossentropyloss
function MAE:new(o, y_pred, y_true)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    o.forward = function (y_pred, y_true) return self:Forward(o, y_pred, y_true) end
    o.backward = function (dvalues, y_true) return self:Backward(o, dvalues, y_true) end
    return self.Super(o, y_pred, y_true)
end

function MAE:Forward(self, y_pred, y_true)
    
end

function MAE:Backward(self, dvalues, y_true)


end

return MAE