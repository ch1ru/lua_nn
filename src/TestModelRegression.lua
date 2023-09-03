local model = require('Model')
require('DenseLayer')
require('AccuracyRegression')
local Relu = require('Relu')
require('TestDataFunctions')
require('DataLoader')
local LinearActivation = require('LinearActivation')
local MSE = require('MSELoss')
local Optimizer = require('Optimizer')
local matrix = require('Matrix')
local Sigmoid= require('Sigmoid')

local X_train, y_train = SinData(0, 1, .001)
local X_val, y_val = SinData(1, 2, 0.001)

local train_dl = DataLoader:new(X_train, y_train)
local val_dl = DataLoader:new(X_val, y_val)

local model = model:new()
model.add(DenseLayer:new(2, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 1))
model.add(LinearActivation:new())

model.set(
    MSE:new(), 
    Optimizer.Adam(0.01, 1e-3), 
    AccuracyRegression:new()
)

model.finalize()

model.train(train_dl, 1000, 5, nil)