require('./Helper')

Tensor = {}

-- Derived class method new
function Tensor:new (o, tensor)
   o = o or {}
   setmetatable(o, self)
   self.__index = self
   self.__mul = function (m1, m2) return Tensor:MatMul(m1, m2) end
   self.__add = function (m, b) return Tensor:AddScalar(m, b) end
   self.tensor = tensor
   return o
end

--Matrix multiplication
function Tensor:MatMul( m1, m2 )
   if #m1[1] ~= #m2 then       --inner matrix-dimensions must agree
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
   
   return Tensor:new(nil, res)
end

--Scalar addition
function Tensor:AddScalar(m, b)
    --swap if in a different order
    if #b ~= 1 and #m == 1 then
        m, b = b, m    
    elseif (#b ~= 1) ~= (#m ~= 1) then --at least one (but not both) needs to be 1D matrix
        --throw error
    end 

    local res = {}
    --cycle outer dimention
    for i = 1, #m do
        res[i] = {}
        --cycle inner dimension
        for j = 1, #m[i] do
            res[i][j] = m[i][j] + b[1][j]
        end
    end
    return Tensor:new(nil, res)
end

--Return size (TODO: allow for rugged tensors)
function Tensor:Shape()
    local isTable = true
    local size = {}
    local currTable = self.tensor
    while isTable do
        local sizeTmp = 0
        for _ in pairs(currTable) do sizeTmp = sizeTmp + 1 end
        table.insert(size, sizeTmp)
        currTable = currTable[1] --go to inner dimension
        if type(currTable) ~= 'table' then
            isTable = false
        end
    end
    return size
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


--Example tensor inputs and weights
local X = {{1,2,3,2.5},{2,5,-1,2},{-1.5,2.7,3.3,-0.8}}
local w = {{0.2,0.5,-0.26}, {0.8,-0.91,-0.27}, {-0.5,0.26,0.17},{1,-0.5,0.87}}
local b = {{2, 3, 0.5}}

local out = Tensor:new(X) * Tensor:new(w)
print(out.tensor + Tensor:new(b))

local btensor = Tensor:new(nil, b)
tprint(btensor:Shape())

