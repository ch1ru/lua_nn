local matrix = require('Matrix')
require('Table')

Accuracy = {}

function Accuracy:new(o)
    local o = o or {}
    setmetatable(o, self)
    o.calculate = function(preds, y_true) return self:Calculate(o, preds, y_true) end
  return o
end

function Accuracy:Calculate(self, preds, y_true)
    
    local pred_t = {}

    local comparisons = self.compare(preds, y_true)

    return comparisons
    
end

return Accuracy