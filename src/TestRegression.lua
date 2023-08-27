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

local X_train, y_train = SinData(0, 1, .001)

local acc_precision = Std(y_train[1]) / 250

local dense1 = DenseLayer:new(1, 64, 0, 5e-4, 0, 5e-4)
local activation1 = Relu:new()

local dense2 = DenseLayer:new(64, 64)
local activation2 = Relu:new()

local dense3 = DenseLayer:new(64, 1)
local activation3 = LinearActivation:new()

local lossFn = MSE:new()
lossFn.rememberTrainableLayers({dense1, dense2, dense3})

local optimizer = Optimizer.Adam(0.01, 1e-3)



for epoch = 1, 100 do
  
  dense1.forward(matrix(X_train))
  activation1.forward(dense1.output)

  dense2.forward(activation1.output)
  activation2.forward(dense2.output)

  dense3.forward(activation2.output)
  activation3.forward(dense3.output)

  local data_loss = lossFn.calculate(activation3.output, y_train)

  local reg_loss = lossFn.regularization_loss(dense1) + lossFn.regularization_loss(dense2) + lossFn.regularization_loss(dense3)

  local loss = data_loss + reg_loss

  local acc = 0.0

  local predictions = activation3.output

  if epoch % 1 == 0 then
    print(string.format("Epoch: %d, Acc: %f, Loss: %f, learning rate: %f", epoch, acc, loss, optimizer.currentlr))
    SaveData('./TrainResults/output.csv', ConvertRegressPredsToCSV(X_train, predictions))
  end

  lossFn.backward(activation3.output, matrix.transpose(y_train))
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