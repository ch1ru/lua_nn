local matrix = require('Matrix')
--ReLU (Rectified Linear Activation Function) returns the input when positive and 0 when negative
-- y = {x x>0, 0 x<=0}

local ReLU = {}

function ReLU:Forward(x)
    for i in x:ipairs() do
       for j = 1, #x[i] do
          if type(x[i][j] == 'table') then
            x[i][j] = math.max(0, x[i][j])
          else
            x[i][j] = math.max(0, x[i][j])
          end
       end
    end
    return x
end

return ReLU