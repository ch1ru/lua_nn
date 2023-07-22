local table = require('Table')
require('Math')
require('TestDataFunctions')
local DenseLayer = require('DenseLayer')
local Relu = require('Relu')
local Softmax = require('Softmax')
require('SoftmaxCrossEntropyLoss')
require('BaseLoss')
local Units = require('UnitTests')
require('Accuracy')
require('Helper')
local Optimizer = require('Optimizer')
local MSE = require('MSELoss')
local matrix = require('Matrix')
local LinearActivation = require('LinearActivation')

--create a sine wave as our training data
local X_train = table.arrange(0, 1, .001)

local y_train = {} --Input training coords
local c = 0
for _, v in ipairs(X_train) do
  table.insert(y_train, math.sin(v * math.pi * 2))
end

X_train = matrix.transpose(matrix({X_train}))

local acc_precision = Std(y_train) / 250

local dense1 = DenseLayer:new(1, 64)
local activation1 = Relu:new()

local dense2 = DenseLayer:new(64, 64)
local activation2 = Relu:new()

local dense3 = DenseLayer:new(64, 1)
local activation3 = LinearActivation:new()

local lossFn = MSE:new()

local optimizer = Optimizer.Adam(0.005, 1e-3)

for i = 1, 1 do
  
  dense1.forward(matrix(X_train))
  activation1.forward(dense1.output)

  dense2.forward(activation1.output)
  activation2.forward(dense2.output)

  dense3.forward(activation2.output)
  activation3.forward(dense3.output)

  local data_loss = lossFn.calculate(activation3.output, y_train)

  --reg loss

  local predictions = activation3.output

  lossFn.backward(activation3.output, y_train)
  activation3.backward(lossFn.dinputs)
  dense3.backward(activation3.dinputs)
  activation2.backward(dense3.dinputs)
  dense2.backward(activation2.dinputs)
  activation1.backward(dense2.dinputs)
  dense1.backward(activation1.dinputs)

  optimizer.PreUpdateParams()
  optimizer.UpdateParams(dense1)
  optimizer.UpdateParams(dense2)
  optimizer.UpdateParams(dense3)
  optimizer.PostUpdateParams()

end