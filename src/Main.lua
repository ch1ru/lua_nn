local matrix = require('Matrix')
require('Table')
require('DenseLayer')
require('Relu')
require('Softmax')
require('CrossEntropyLoss')

--create a sine wave as our training data
local sinX = table.arrange(1, 10, .1)
local train_set = {}
for _, v in ipairs(sinX) do
    table.insert(train_set, {v, math.sin(v)})
end
train_set = matrix:new(train_set)

local dense1 = DenseLayer:new(nil, 2, 3)
local dense2 = DenseLayer:new(nil, 3, 3)

local X = dense1:Forward(train_set)
print(X)
X = ReLU(X)
X = dense2:Forward(X)
X = Softmax(X)
--print(X)
