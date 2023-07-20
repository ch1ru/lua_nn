local matrix = require('Matrix')
local table = require('Table')
require('Helper')

--Example tensor inputs and weights
local X = matrix:new({{1,2,3,2.5},{2,5,-1,2},{-1.5,2.7,3.3,-0.8}})
local w = matrix:new({{0.2,0.5,-0.26}, {0.8,-0.91,-0.27}, {-0.5,0.26,0.17},{1,-0.5,0.87}})
local b = matrix:new({{2,3,0.5},{2,3,0.5},{2,3,0.5}})

local xw_b = X * w + b

--create a sine wave as our training data
--local X = table.arrange(1, 10, .1)

--local X = {} --Input training coords
--for i = 1, 10, 0.1 do
--    table.insert(X, {i, math.sin(i)})
--end

---local y = {}
--for _, v in ipairs(X) do
  --  if(v[2] > 0) then
    --    table.insert(y, 1)
   -- else
     --   table.insert(y, 0)
    --end
--end

--X = matrix(X)
--y = matrix({y})

-------------------------------------
-------------Test data---------------


--create 2 circles 1 in the other
-- y^2 + x^2 = 25
-- y^2 + x^2 = 100

--first generate random points between -15 and 15
local X = {}
for i = 1, 2000 do
    local signx, signy
    local randx = math.random()
    local randy = math.random()
    if randx < 0.5 then
      signx = -1
    else
      signx = 1
    end
    if randy < 0.5 then
      signy = -1
    else
       signy = 1
    end
    table.insert(X, {math.random() * 15 * signx, math.random() * 15 * signy})
end

--next, classify points
local y = {}
for _, v in ipairs(X) do
    local squares = v[2] * v[2] + v[1] * v[1]
    if squares > 10 and squares < 50 then
      table.insert(y, 0)
    elseif squares > 50 and squares < 115 then
      table.insert(y, 1)
    elseif squares > 0 and squares < 10 then
      table.insert(y, 2)
    else
      table.insert(y, 3)
    end
end


