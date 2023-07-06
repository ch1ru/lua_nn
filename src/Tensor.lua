--Example tensor inputs and weights
local X = {{1,2,3,2.5},{2,5,-1,2},{-1.5,2.7,3.3,-0.8}}
local w = {{0.2,0.5,-0.26}, {0.8,-0.91,-0.27}, {-0.5,0.26,0.17},{1,-0.5,0.87}}

Tensor = {}

-- Derived class method new
function Tensor:new (o, tensor)
   o = o or {}
   setmetatable(o, self)
   self.__index = self
   self.__mul = function (m1, m2) return Tensor:MatMul(m1, m2) end
   self.__add = function (m1, m2) return Tensor:MatAdd(m1, m2) end
   self.tensor = tensor
   return o
end

--Matrix multiplication
function Tensor:MatMul( m1, m2 )
   if #m1[1] ~= #m2 then       -- inner matrix-dimensions must agree
       return nil      
   end 

   local res = {}
   
   for i = 1, #m1 do
       res[i] = {}
       for j = 1, #m2[1] do
           res[i][j] = 0
           for k = 1, #m2 do
               res[i][j] = res[i][j] + m1[i][k] * m2[k][j]
           end
       end
   end
   
   res = Tensor:new(nil, res)
   return res
end

--Matrix addition
function Tensor:MatAdd(m1, m2)
    return nil
end

--Displays the tensor
function Tensor:__tostring()
    local str = ""
    for index, data in ipairs(self.tensor) do
        for key, value in pairs(data) do
            str = str .. value .. '\t'
        end
        str = str .. '\n'
    end
    return str
end

--Transpose matrix
function Tensor:Transpose()
    return nil
end

local out = Tensor:new(X) * Tensor:new(w)
print(out)
