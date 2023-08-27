local model = require('Model')
require('DenseLayer')
require('AccuracyRegression')
local Relu = require('Relu')
require('TestDataFunctions')
local LinearActivation = require('LinearActivation')
local MSE = require('MSELoss')
local Optimizer = require('Optimizer')

local X_train, y_train = SinData(0, 1, .001)

local model = model:new()
model.add(DenseLayer:new(1, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 1))
model.add(LinearActivation:new())

model.set(
    MSE:new(), 
    Optimizer.Adam(0.005, 1e-3), 
    AccuracyRegression:new())

model.finalize()

model.train(X_train, y_train, 1000, 5, nil)