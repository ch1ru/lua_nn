AccuracyCategorical = { y_pred = {}, y_true = {} }
require('Accuracy')
local table = require('Table')
local matrix = require('Matrix')

AccuracyCategorical.Super = function (self) return Accuracy:new(self) end

function AccuracyCategorical:new(binary)
    local o = {}
    setmetatable(o, self)
    self.__index = self
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

    -- local pred_t = {}

    -- for i = 1, #preds do
    --     for j = 1, #preds[i] do
    --         if table.max(preds[i])[1] == preds[i][j] then
    --             table.insert(pred_t, j-1)
    --         end
    --     end
    -- end

    -- local correct_counter = 0
    -- for i = 1, #y_true[1] do
    --     if y_true[1][i] == pred_t[i] then
    --         correct_counter = correct_counter + 1
    --     end
    -- end

    -- local mean = correct_counter / #y_true[1]

    -- return mean

    return {0.0}
end

return AccuracyCategorical