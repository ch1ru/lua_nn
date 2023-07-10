require('Helper')

local matrix = require 'Matrix'
local table = require "LuaTable"

Tensor = { tensor = {} }

-- Derived class method new
function Tensor:new (o, tensor)
   local o = o or {}
   setmetatable(o, self)
   self.__index = self
   o.tensor = tensor
   self.__mul = function (m1, m2) return Tensor:MatMul(m1.tensor, m2.tensor) end
   self.__add = function (m1, b) return Tensor:AddScalar(m1.tensor, b.tensor) end
   self.__sub = function (m1, m2) return Tensor:MatSub(m1.tensor, m2.tensor) end
   self.__div = function (m1, m2) return Tensor:MatDiv(m1.tensor, m2.tensor) end
   return o
end

--Matrix multiplication
function Tensor:MatMul( m1, m2 )

   --if #m1.tensor ~= #m2.tensor then       --inner matrix-dimensions must agree
   --    return nil      
   --end 

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
    --cycle outer dimension
    for i = 1, #m do
        res[i] = {}
        --cycle inner dimension
        for j = 1, #m[i] do
            res[i][j] = m[i][j] + b[1][j]
        end
    end
    return Tensor:new(nil, res)
end

function Tensor:MatDiv(m1, m2)
    
end

function Tensor:MatSub(m1, m2)

    local res = {}
    --cycle outer dimension
    for i = 1, #m1 do
        res[i] = {}
        --cycle inner dimension
        for j = 1, #m2 do
            res[i][j] = m1[i][j] - m2[1][j]
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
        for key, value in ipairs(data) do
            str = str .. value .. '\t'
        end
        str = str .. '\n'
    end
    return str
end

--Converts a tensor of any dimension into a single dimension
function ConvertTo1D(tensor)
    local newTable = {}
    for _, v in ipairs(tensor) do
        if type(v) == 'table' then
            local v = ConvertTo1D(v)
            for _, w in ipairs(v) do
                table.insert(newTable, w)
            end
        else
            table.insert(newTable, v)
        end
    end
    return newTable
end

--Create multi-dimensional table from a single table
function ResizeTable(singleTable, newShape)
    local newTable = ConvertTo1D(singleTable)
    
    for i = #newShape, 2, -1 do
        newTable = table.explode(newTable, newShape[i])
    end
    return newTable
end


--Transpose matrix
function Tensor:Transpose(newShape)
    local originalTensor = self.tensor
    local singleTensor = ConvertTo1D(originalTensor)
    local newTensor = singleTensor
    
    for i = #newShape, 2, -1 do
        newTensor = table.explode(newTensor, newShape[i])
    end
    return Tensor:new(nil, newTensor)
end


function Tensor:Ones(size)
    size = size or self:Shape()
    return Tensor:new(nil, table.ones(size))
end

function Tensor:Zeros(size)
    size = size or self:Shape()
    return Tensor:new(nil,table.zeros(size))
end

--Sum values in table TODO: add options for axis and keepdims
function Tensor:Sum(x)
    local sum = 0
    local shape = x:Shape()
    print(tprint(shape))
    local singleTable = ConvertTo1D(x.tensor)
    for _, v in ipairs(singleTable) do
      sum = sum + v
    end
    return Tensor:new(nil, ResizeTable(sum, shape))
end

function Tensor:Exp(x)
    local shape = x:Shape()
    local t = {}
    local singleTensor = ConvertTo1D(x.tensor)
    for _, v in ipairs(singleTensor) do
        table.insert(t, math.exp(v))
    end
    --convert back to original shape
    return Tensor:new(nil, ResizeTable(t, shape))
end

function Tensor:Max(x)
    local maxTables = {}
    for _, v in ipairs(x.tensor) do
        table.insert(maxTables, table.max(v))
    end
    return Tensor:new(nil, {maxTables})
end

function Tensor:Min(x)
    local minTables = {}
    for _, v in ipairs(x.tensor) do
        table.insert(minTables, table.min(v))
    end
    return Tensor:new(nil, {minTables})
end

--Example tensor inputs and weights
local X = {{1,2,3,2.5},{2,5,-1,2},{-1.5,2.7,3.3,-0.8}}
local w = {{0.2,0.5,-0.26}, {0.8,-0.91,-0.27}, {-0.5,0.26,0.17},{1,-0.5,0.87}}
local b = {{2, 3, 0.5}}

--local Xtensor = Tensor:new(nil, X)
--local wtensor = Tensor:new(nil, w)
--local btensor = Tensor:new(nil, b)

--local xw = Xtensor * wtensor
--print(xw)
--local xw_b = xw - btensor
--print(xw_b)
--print("Original Tensor")
--local x = Tensor:new(nil, X)
--print(x)
--print("shape")
--print(tprint(x:Shape()))
--print()
--print(out.tensor + Tensor:new(b))

--local btensor = Tensor:new(nil, b)
--tprint(btensor:Shape())
--print()
--print("new Tensor")
--local xt = x:Transpose({4, 3})
--print(xt)
--print("new shape")
--print(tprint(xt:Shape()))

return Tensor