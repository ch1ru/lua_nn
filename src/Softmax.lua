require('Table')
require('Helper')

local matrix = require('Matrix')

--Softmax gives a confidence level for unnormalized inputs. This can be very useful for classification systems
--For example, the output [128, 9, 44] provides no context. However when softmax is applied it could give
--a probability distribution [0.15, 0.85] to express confidence between identification of two classes.

--The scary equation for softmax is:
-- s(i, j) = e^z(i,j) / Σe^z(i, j)

local Softmax = {}

function Softmax:Forward(x)

    local exp_values = table.exp(x - matrix(table.max(x)))
    local probabilities = MatDivByRowOrCol(matrix(exp_values), matrix(table.sumT(exp_values)))
    return probabilities
end

return Softmax






