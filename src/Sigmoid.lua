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
    local t = {}
    for _, v in ipairs(x) do
        table.insert(t, 1 / 1 + math.exp(-v))
    end
    self.output = matrix(t)
    return self.output
end

function Sigmoid:Backward(self, dvalues)

end

return Sigmoid