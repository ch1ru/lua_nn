local matrix = require('Matrix')
--ReLU (Rectified Linear Activation Function) returns the input when positive and 0 when negative
-- y = {x x>0, 0 x<=0}

local ReLU = {}

function ReLU:new(o)
  o = o or {}
  setmetatable(o, self)
  o.forward = function (x) return self:Forward(o, x) end
  o.backward = function (dvalues) return self:Backward(o, dvalues) end
  o.predictions = function (outputs) return self:Predictions(o, outputs) end
  return o
end

function ReLU:Forward(self, x)
  self.inputs = x
  local newM = matrix:new(x:rows(), x:columns())
    for i in x:ipairs() do
       for j = 1, #x[i] do
          if type(x[i][j] == 'table') then
            newM[i][j] = math.max(0, x[i][j])
          else
            newM[i][j] = math.max(0, x[i][j])
          end
       end
    end
    self.output = newM
    return self.output
end

function ReLU:Backward(self, dvalues)
  self.dinputs = dvalues
  
  --give a zero gradient where inputs where negative
  for i = 1, self.inputs:rows() do
    for j = 1, self.inputs:columns() do
      if self.inputs[i][j] < 0 then
        matrix.setelement(self.dinputs, i, j, 0)
      end
    end
  end
  return self.dinputs
end

function ReLU:Predictions(self, outputs)
  return outputs
end

return ReLU