local matrix = require('Matrix')

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




