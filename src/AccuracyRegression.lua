AccuracyRegression = { y_pred = {}, y_true = {} }
require('Accuracy')
local table = require('Table')
local matrix = require('Matrix')

AccuracyRegression.Super = function (self) return Accuracy:new(self) end

function AccuracyRegression:new()
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.precision = nil
    --functions
    o.compare = function (preds, y_true) return self:Compare(o, preds, y_true) end
    o.init = function (y, reinit) return self:Init(o, y, reinit) end
    return self.Super(o)
end

function AccuracyRegression:Init(self, y, reinit)
    if self.precision == nil or reinit then
        self.precision = Std(y[1]) / 250
    end
end

function AccuracyRegression:Compare(self, preds, y_true)
    --local abs = matrix.abs(matrix(preds) - matrix(y_true))
end