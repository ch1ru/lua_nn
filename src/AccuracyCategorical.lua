AccuracyCategorical = { y_pred = {}, y_true = {} }
require('Accuracy')
local table = require('Table')
local matrix = require('Matrix')

AccuracyCategorical.Super = function (self) return Accuracy:new(self) end

function AccuracyCategorical:new(binary)
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.name = "accuracy_categorical"
    o.binary = binary
    --functions
    o.compare = function (preds, y_true) return self:Compare(o, preds, y_true) end
    o.init = function (y, reinit) return self:Init(o, y, reinit) end
    return self.Super(o)
end

function AccuracyCategorical:Init(self, y, reinit)
    --pass
end

function AccuracyCategorical:Compare(self, preds, y_true)

    local pred_true = {}

    --multiclass predictions
    --do argmax if binary and contains multiple preds (softmax)
    if not self.binary and type(preds[1]) == 'table' then
        for i = 1, #preds do
            preds[i] = table.max(preds[i])[1]
        end
    end

    --binary predictions
    for i = 1, #preds do
        if preds[i] == y_true[1][i] then
            table.insert(pred_true, 1)
        else
            table.insert(pred_true, 0)
        end
    end

    
    return pred_true
end

return AccuracyCategorical