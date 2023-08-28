local model = require('Model')
require('DenseLayer')
require('AccuracyRegression')
local Relu = require('Relu')
require('TestDataFunctions')
local matrix = require('Matrix')
local LinearActivation = require('LinearActivation')
local MSE = require('MSELoss')
local Optimizer = require('Optimizer')
local Sigmoid = require('Sigmoid')
local BinaryCrossEntropyLoss = require('BinaryCrossEntropyLoss')
local table = require('table')
local X, y = GenerateBinaryClasses(100)

local dense1 = DenseLayer:new(2, 64)
local activation1 = Relu:new()
local dense2 = DenseLayer:new(64, 1)
local activation2 = Sigmoid:new()
local loss_fn = BinaryCrossEntropyLoss:new()
local optimizer = Adam:new(0.001, 5e-7)

for epoch = 1, 2000 do
    dense1.forward(X)
    activation1.forward(dense1.output) 
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    local data_loss, _ = loss_fn.calculate(activation2.output, y)
    
    local loss = data_loss
    local acc = 0.0
    if epoch % 10 == 0 then
        print(string.format("Epoch: %d, Acc: %f, Loss: %f, learning rate: %f", epoch, acc, loss, optimizer.currentlr))
    end

    local preds = {}
    for i = 1, activation2.output:rows() do
        for j = 1, activation2.output:columns() do
            if activation2.output[i][j] > 0.5 then
                table.insert(preds, 1)
            else
                table.insert(preds, 0)
            end
        end
    end
    
    loss_fn.backward(activation2.output, y)
    activation2.backward(loss_fn.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.PreUpdateParams()
    optimizer.UpdateParams(dense1)
    optimizer.UpdateParams(dense2)
    optimizer.PostUpdateParams()
end