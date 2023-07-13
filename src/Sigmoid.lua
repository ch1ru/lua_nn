--The sigmoid activation function normalizes input values between 0 and 1
--y = 1 / 1 + e^-x

local matrix = require('Matrix')

local Sigmoid = {}

function Sigmoid:Forward(x)
    local t = {}
    for _, v in ipairs(x) do
        table.insert(t, 1 / 1 + math.exp(-v))
    end
    return matrix(t)
end

return Sigmoid