BinaryCrossEntropyLoss = { y_pred = {}, y_true = {} }
require('BaseLoss')
local table = require('Table')
local matrix = require('Matrix')

--Super method for calling parent constructor
BinaryCrossEntropyLoss.Super = function (self, y_pred, y_true) return BaseLoss:new(self) end

-- Derived class method crossentropyloss
function BinaryCrossEntropyLoss:new()
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.name = "binary_crossentropy_loss"
    o.forward = function (y_pred, y_true) return self:Forward(o, y_pred, y_true) end
    o.backward = function (dvalues, y_true) return self:Backward(o, dvalues, y_true) end
    return self.Super(o)
end

function BinaryCrossEntropyLoss:Forward(self, y_pred, y_true)
    --Weight samples to not affect the mean too much
    local y_pred_clipped = table.clip(y_pred, 1e-7, 1 - 1e-7)
    local y_true_one_minus = {}
    local y_pred_clipped_one_minus = {}
    local alpha = {}
    for i = 1, y_true:rows() do
        for j = 1, y_true:columns() do
            table.insert(y_true_one_minus, 1 - y_true[i][j])
            table.insert(y_pred_clipped_one_minus, math.log(1 - (y_pred_clipped)[i * j][1]))
            table.insert(alpha, y_true[i][j] * table.log(y_pred_clipped)[i * j][1])
        end
    end

    local beta = {}
    for i = 1, #y_pred_clipped_one_minus do
        table.insert(beta, y_true_one_minus[i] * y_pred_clipped_one_minus[i])
    end
    
    local sample_losses = {} 
    for i = 1, #alpha do
        table.insert(sample_losses, alpha[i] + beta[i])
    end
    
    return table.makeNegative(sample_losses)
    
end

function BinaryCrossEntropyLoss:Backward(self, dvalues, y_true)
    local samples = dvalues:rows()
    local outputs = dvalues:columns()
    local dvalues_clipped = table.clip(dvalues, math.exp(-7), 1 - math.exp(-7))

    local alpha = {}
    local beta = {}
    local gamma = {}
    local dinputs = {}
    local dinputs_samples = {}

    for i = 1, #dvalues_clipped do
        table.insert(alpha, y_true[1][i] / dvalues_clipped[i][1])
        table.insert(beta, (1 - y_true[1][i]) / (1 - dvalues_clipped[i][1]))
    end

    for i = 1, #alpha do
        table.insert(gamma, alpha[i] - beta[i])
    end

    for i = 1, #gamma do
        table.insert(dinputs, (-1 * gamma[i]) / outputs)
    end

    for i = 1, #dinputs do
        table.insert(dinputs_samples, dinputs[i] / samples)
    end
    
    self.dinputs = matrix.transpose(matrix:new({dinputs_samples}))
end

return BinaryCrossEntropyLoss