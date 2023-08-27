--The sigmoid activation function normalizes input values between 0 and 1
--y = 1 / 1 + e^-x

local matrix = require('Matrix')

local Sigmoid = {}

function Sigmoid:new(o)
    o = o or {}
    setmetatable(o, self)
    o.forward = function (x) return self:Forward(o, x) end
    o.backward = function (dvalues) return self:Backward(o, dvalues) end
    return o
end

function Sigmoid:Forward(self, x)
    local m3 = matrix:new({x:rows(), x:columns()})
    self.inputs = x
    for i = 1, x:rows() do
        m3[i] = {}
        for j = 1, x:columns() do
            m3[i][j] = 1 / (1 + math.exp(x[i][j]))
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

return Sigmoid