require('lua_nn/src/?.lua')

--Softmax gives a confidence level for unnormalized inputs. This can be very useful for classification systems
--For example, the output [128, 9, 44] provides no context. However when softmax is applied it could give
--a probability distribution [0.15, 0.85] to express confidence between identification of two classes.

--The big scary equation for softmax is:
-- s(i, j) = e^z(i,j) / Σe^z(i, j)

function Softmax(x)
    local t = {}
    local exp_values = {}
    for _, v in ipairs(x.tensor) do
        table.insert(exp_values, math.exp(v))
    end
    local norm_base = Tensor:sum(exp_values, 1, true)
    local norm_values = {}
    for value in exp_values do
        table.insert(norm_values, value / norm_base)
    end
end