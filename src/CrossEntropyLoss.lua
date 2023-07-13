
CrossEntropyLoss = { y_pred = {}, y_true = {} }
require('BaseLoss')
require('Table')

local matrix = require('Matrix')

--Super method for calling parent constructor
CrossEntropyLoss.Super = function (self, y_pred, y_true) return BaseLoss:new(self, y_pred, y_true) end

-- Derived class method crossentropyloss
function CrossEntropyLoss:new(o, y_pred, y_true)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    o.Forward = function (y_pred, y_true) return CrossEntropyLoss:Forward(y_pred, y_true) end
    return self.Super(o, y_pred, y_true)
end

function CrossEntropyLoss:Forward (y_pred, y_true)
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

return CrossEntropyLoss

