local matrix = require('Matrix')
--ReLU (Rectified Linear Activation Function) returns the input when positive and 0 when negative
-- y = {x x>0, 0 x<=0}

function ReLU(x)
    for i= 1, #x do

       for j = 1, #x[i] do
          if type(x[i][j] == 'table') then
            x[i][j] = table(math.max(0, x[i][j][1]))
          else
            x[i][j] = math.max(0, x[i][j])
          end
       end
        
    end
    return x
end