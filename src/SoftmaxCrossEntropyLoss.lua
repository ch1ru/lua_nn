SoftmaxCrossEntropyLoss = { }
require('BaseLoss')
require('Table')

local Softmax = require('Softmax')
local CrossEntropyLoss = require('CrossEntropyLoss')
local matrix = require('Matrix')

function SoftmaxCrossEntropyLoss:new(o)
    o = o or {}
    setmetatable(o, self)
    o.activation = Softmax:new()
    o.loss = CrossEntropyLoss:new()
    o.forward = function (inputs, y_true) return self:Forward(o, inputs, y_true) end
    o.backward = function (dvalues, y_true) return self:Backward(o, dvalues, y_true) end
    return o
end

function SoftmaxCrossEntropyLoss:Forward(self, inputs, y_true)
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)
end

function SoftmaxCrossEntropyLoss:Backward(self, dvalues, y_true)
    local samples = dvalues:columns() * dvalues:rows()
    if y_true:rows() == 2 then
        --one hot encoded
    end

    self.dinputs = dvalues
    for i = 1, self.dinputs:rows() do
        dvalues[i][y_true[1][i]+1] = dvalues[i][y_true[1][i]+1] - 1
    end
    
    self.dinputs = matrix.divnum(self.dinputs, samples)
    return self.dinputs

end