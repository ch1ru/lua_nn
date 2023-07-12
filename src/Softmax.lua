require('Table')
require('Helper')

local matrix = require('Matrix')

--Softmax gives a confidence level for unnormalized inputs. This can be very useful for classification systems
--For example, the output [128, 9, 44] provides no context. However when softmax is applied it could give
--a probability distribution [0.15, 0.85] to express confidence between identification of two classes.

--The big scary equation for softmax is:
-- s(i, j) = e^z(i,j) / Σe^z(i, j)

function Softmax(x)
    local t1 = {{2,5,8},{3,3,3},{4,5,2},{5,1,3},{6,8,99},{76,8,1},{8,8,3}}
    local t2 = table({1,2,3,4,5,6,7,8,9})

    local exp_values = table.exp(X - matrix(table.max(X)))
    local probabilities = matrix(exp_values):divnum(matrix(table.sumT(exp_values)))
    return probabilities
end






