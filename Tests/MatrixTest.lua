--Example tensor inputs and weights
local X = {{1,2,3,2.5},{2,5,-1,2},{-1.5,2.7,3.3,-0.8}}
local w = {{0.2,0.5,-0.26}, {0.8,-0.91,-0.27}, {-0.5,0.26,0.17},{1,-0.5,0.87}}
local b = table.duplicate({2,3,0.5},3)

local Xtensor = matrix:new(X)
local wtensor = matrix:new(w)
local btensor = matrix:new(b)

local xw_b = Xtensor * wtensor + btensor