local model = require('Model')
require('DenseLayer')
local Relu = require('Relu')
require('TestDataFunctions')
local Optimizer = require('Optimizer')
local Sigmoid = require('Sigmoid')
local BinaryCrossEntropyLoss = require('BinaryCrossEntropyLoss')
local AccuracyCategorical = require('AccuracyCategorical')
local Softmax = require('Softmax')
require('DataLoader')

local X_train, y_train = GenerateBullseye(250)
local X_val, y_val = GenerateBullseye(20)

local train_dl = DataLoader:new(X_train, y_train)
local val_dl = DataLoader:new(X_val, y_val)

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

model.train(train_dl, 1000, 5)




