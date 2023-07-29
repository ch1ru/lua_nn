
CrossEntropyLoss = { y_pred = {}, y_true = {} }
require('BaseLoss')
local table = require('Table')

local matrix = require('Matrix')

--Super method for calling parent constructor
CrossEntropyLoss.Super = function (self, y_pred, y_true) return BaseLoss:new(self) end

-- Derived class method crossentropyloss
function CrossEntropyLoss:new()
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.forward = function (y_pred, y_true) return self:Forward(o, y_pred, y_true) end
    o.backward = function (dvalues, y_true) return self:Backward(o, dvalues, y_true) end
    return self.Super(o)
end

function CrossEntropyLoss:Forward (self, y_pred, y_true)
    local samples = y_pred:size()
    
    --Weight samples to not affect the mean too much
    local y_pred_clipped = table.clip(y_pred, math.exp(-7), 1 - math.exp(-7))
    
    local correctConfidences = {}
    if matrix(y_true):rows() == 1 then
        for i = 1, #y_pred_clipped do
            --io.write(y_pred_clipped[i][1] .. "   " .. y_pred_clipped[i][2] .. "   " .. y_true[1][i] .. "   " .. y_pred_clipped[i][y_true[1][i]+1])
            --print()
            table.insert(correctConfidences, y_pred_clipped[i][y_true[1][i]+1])

        end
    elseif matrix(y_true):columns() == 1 then 
        print("Predictions in the wrong shape, Perhaps resize from column to row vector?") 
        return nil
    else
        --add for one hot coded labels
    end
    local logLiklihoods = table.log(correctConfidences)
    local negativeLogLiklihoods = table.makeNegative(logLiklihoods)
    return negativeLogLiklihoods
end

function CrossEntropyLoss:Backward(self, dvalues, y_true)
    local samples = dvalues:columns() * dvalues:rows()
    local labels = dvalues:columns()
    self.dinputs = matrix(table.makeNegative(y_true)) / dvalues
    self.dinputs = self.dinputs / samples
end

return CrossEntropyLoss

