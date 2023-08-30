local model = require('Model')
require('DenseLayer')
local Relu = require('Relu')
require('TestDataFunctions')
local Optimizer = require('Optimizer')
local Sigmoid = require('Sigmoid')
local BinaryCrossEntropyLoss = require('BinaryCrossEntropyLoss')
local AccuracyCategorical = require('AccuracyCategorical')
local DataLoader = require('DataLoader')

local X_train, y_train = GenerateBinaryClasses(80)
local X_val, y_val = GenerateBinaryClasses(2)

local train_dl = DataLoader:new(X_train, y_train, 20)
local val_dl = DataLoader:new(X_val, y_val)



local model = model:new()

model.add(DenseLayer:new(2, 64))
model.add(Relu:new())
model.add(DenseLayer:new(64, 1))
model.add(Sigmoid:new())

model.set(
    BinaryCrossEntropyLoss:new(),
    Optimizer.Adam(0.005, 1e-3),
    AccuracyCategorical:new(true)
)



model.finalize()


model.train(train_dl, 2000, 5)