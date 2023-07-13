local matrix = require('Matrix')
require('Table')
require('DenseLayer')
local Relu = require('Relu')
local softmax = require('Softmax')
require('CrossEntropyLoss')
require('BaseLoss')

--create a sine wave as our training data
--local X = table.arrange(1, 10, .1)

local X = {} --Input training coords
for i = 1, 10, 0.1 do
    table.insert(X, {i, math.sin(i)})
end

local y = {}
for _, v in ipairs(X) do
    if(v[2] > 0) then
        table.insert(y, 1)
    else
        table.insert(y, 0)
    end
end

X = matrix(X)
y = matrix({y})

local dense1 = DenseLayer:new(nil, 2, 3)
local dense2 = DenseLayer:new(nil, 3, 2)

X = dense1:Forward(X)
X = Relu:Forward(X)
X = dense2:Forward(X)
X = softmax:Forward(X)

local loss_fn = CrossEntropyLoss:new(nil, X, y)



local losses = loss_fn:Calculate(X, y)
print(losses)
