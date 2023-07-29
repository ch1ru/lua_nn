MSE = { y_pred = {}, y_true = {} }

require('BaseLoss')
require('Table')

local matrix = require('Matrix')

--Super method for calling parent constructor
MSE.Super = function (self) return BaseLoss:new(self) end

-- Derived class method crossentropyloss
function MSE:new()
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.forward = function (y_pred, y_true) return self:Forward(o, y_pred, y_true) end
    o.backward = function (dvalues, y_true) return self:Backward(o, dvalues, y_true) end
    return self.Super(o)
end

function MSE:Forward(self, y_pred, y_true)

    local sampleLosses = y_true - matrix.transpose(y_pred)

    local sampleLossesTbl = {}
    for i = 1, #sampleLosses[1] do
        matrix.setelement(sampleLosses, 1, i, sampleLosses[1][i] * sampleLosses[1][i])
        
    end

    --TODO add mean for each output sample
    for i = 1, #sampleLosses[1] do
        table.insert(sampleLossesTbl, sampleLosses[1][i])
    end

    return sampleLossesTbl
end

function MSE:Backward(self, dvalues, y_true)
    local samples = dvalues:columns() * dvalues:rows()
    local outputs = dvalues:columns()
    self.dinputs = matrix.divnum(
        matrix.mulnum((y_true - dvalues), -2), outputs)
    self.dinputs = matrix.divnum(self.dinputs, samples)
    self.dinputs = self.dinputs
end

return MSE