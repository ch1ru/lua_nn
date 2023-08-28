--The sigmoid activation function normalizes input values between 0 and 1
--y = 1 / 1 + e^-x

local matrix = require('Matrix')

local Sigmoid = {}

function Sigmoid:new(o)
    o = o or {}
    setmetatable(o, self)
    o.forward = function (x) return self:Forward(o, x) end
    o.backward = function (dvalues) return self:Backward(o, dvalues) end
    o.predictions = function(outputs) return self:Predictions(o, outputs) end
    return o
end

function Sigmoid:Forward(self, x)
    local m3 = matrix:new({x:rows(), x:columns()})
    self.inputs = x
    for i = 1, x:rows() do
        m3[i] = {}
        for j = 1, x:columns() do
            m3[i][j] = 1 / (1 + math.exp(-x[i][j]))
        end
    end
    self.output = m3
    return self.output
end

function Sigmoid:Backward(self, dvalues)
    local output_one_minus = matrix({self.output:rows(), self.output:columns()})
    for i = 1, self.output:rows() do
        output_one_minus[i] = {}
        for j = 1, self.output:columns() do
            output_one_minus[i][j] = 1 - self.output[i][j]
        end
    end
    self.dinputs = dvalues * output_one_minus * self.output
end

function Sigmoid:Predictions(self, outputs) 
    local preds = {}
    for i = 1, outputs:rows() do
        for j = 1, outputs:columns() do
            if outputs[i][j] > 0.5 then
                table.insert(preds, 1)
            else
                table.insert(preds, 0)
            end
        end
    end
end

return Sigmoid