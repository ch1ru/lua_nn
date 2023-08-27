AccuracyClassifier = { y_pred = {}, y_true = {} }
require('Accuracy')
local table = require('Table')
local matrix = require('Matrix')

AccuracyClassifier.Super = function (self) return Accuracy:new(self) end

function AccuracyClassifier:new()
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.precision = nil
    --functions
    o.compare = function (preds, y_true) return self:Compare(o, preds, y_true) end
    return self.Super(o)
end

function AccuracyClassifier:Init(self, y, reinit)
    if self.precision == nil or reinit then
        self.precision = Std(y[1]) / 250
    end
end

function AccuracyClassifier:Compare(self, preds, y_true)

    local pred_t = {}

    for i = 1, #preds do
        for j = 1, #preds[i] do
            if table.max(preds[i])[1] == preds[i][j] then
                table.insert(pred_t, j-1)
            end
        end
    end

    local correct_counter = 0
    for i = 1, #y_true[1] do
        if y_true[1][i] == pred_t[i] then
            correct_counter = correct_counter + 1
        end
    end

    local mean = correct_counter / #y_true[1]

    return mean
end