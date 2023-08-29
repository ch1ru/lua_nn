local model = require('Model')
require('DenseLayer')
local Relu = require('Relu')
require('TestDataFunctions')
local Optimizer = require('Optimizer')
local Sigmoid = require('Sigmoid')
local BinaryCrossEntropyLoss = require('BinaryCrossEntropyLoss')
local AccuracyCategorical = require('AccuracyCategorical')
local Softmax = require('Softmax')

local X_train, y_train = GenerateBullseye(100)
local X_val, y_val = GenerateBullseye(20)

local model = model:new()

model.add(DenseLayer:new(2, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 4))
model.add(Softmax:new())

model.set(
    CrossEntropyLoss:new(),
    Optimizer.Adam(0.005, 1e-3),
    AccuracyCategorical:new()
)

model.finalize()

model.train(X_train, y_train, 1000, 5, {X_val, y_val})




