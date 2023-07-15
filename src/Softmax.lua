require('Table')
require('Helper')

local matrix = require('Matrix')

--Softmax gives a confidence level for unnormalized inputs. This can be very useful for classification systems
--For example, the output [128, 9, 44] provides no context. However when softmax is applied it could give
--a probability distribution [0.15, 0.85] to express confidence between identification of two classes.

--The scary equation for softmax is:
-- s(i, j) = e^z(i,j) / Σe^z(i, j)

local Softmax = {}

function Softmax:new(o)
    o = o or {}
    setmetatable(o, self)
    o.forward = function (x) return self:Forward(o, x) end
    o.backward = function (dvalues) return self:Backward(o, dvalues) end
    return o
end

function Softmax:Forward(self, x)
    self.inputs = x
    local exp_values = table.exp(x - matrix(table.max(x)))
    local probabilities = MatDivByRowOrCol(matrix(exp_values), matrix(table.sumT(exp_values)))
    self.output = probabilities
    return probabilities
end

function Softmax:Backward(self, dvalues)

end

return Softmax






