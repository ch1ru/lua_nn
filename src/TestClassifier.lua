local matrix = require('Matrix')
require('Table')
require('TestDataFunctions')
local DenseLayer = require('DenseLayer')
local Relu = require('Relu')
local Softmax = require('Softmax')
require('SoftmaxCrossEntropyLoss')
require('BaseLoss')
local Units = require('UnitTests')
require('Accuracy')
require('AccuracyClassifier')
local Optimizer = require('Optimizer')

--layers
local dense1 = DenseLayer:new(2, 64, 0, 5e-4, 0, 5e-4)
local dense2 = DenseLayer:new(64, 64, 0, 5e-4, 0, 5e-4)
local dense3 = DenseLayer:new(64, 4, 0, 5e-4, 0, 5e-4)

--activation functions
local activation1 = Relu:new()
local activation2 = Relu:new()
local loss_activation = SoftmaxCrossEntropyLoss:new()
loss_activation.loss.rememberTrainableLayers({dense1, dense2, dense3})
local softmax = Softmax:new()

--optimizer
local optimizer = Optimizer.SGD(.01, 0.0005, 0.9)
local accuracy = AccuracyClassifier:new()

--generate training and test data
local X_train, y_train = GenerateBullseye(2000)
local X_test, y_test

--metrics trackers
local loss_tracker = {}
local acc_tracker = {}
local epoch_tracker = {}

--create training loop
for epoch = 1, 3000 do

    table.insert(epoch_tracker, epoch)

    dense1.forward(X_train)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)

    local loss = loss_activation.forward(dense3.output, y_train)
    table.insert(loss_tracker, loss)

    local acc = accuracy.calculate(loss_activation.output, y_train)
    table.insert(acc_tracker, acc)

    if epoch % 5 == 0 then
        print(string.format("Epoch: %d, Acc: %f, Loss: %f, learning rate: %f", epoch, acc, loss, optimizer.currentlr))
    end

    if epoch % 100 == 0 then
        print("testing...")

        X_test, y_test = GenerateBullseye(2000)

        dense1.forward(X_test)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        dense3.forward(activation2.output)

        local data_loss = loss_activation.forward(dense3.output, y_test)
        local reg_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2) + loss_activation.loss.regularization_loss(dense3)
        local loss = data_loss + reg_loss

        local acc = accuracy.calculate(loss_activation.output, y_test)

        print(string.format("Epoch: %d, Acc: %f, Loss: %f, learning rate: %f", epoch, acc, loss, optimizer.currentlr))

        local preds = softmax.predictions(dense3.output)
        SaveData('./TrainResults/output.csv', ConvertPredsToCSV(X_test, preds))
    end

    loss_activation.backward(loss_activation.output, y_train)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense3.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.PreUpdateParams()
    optimizer.UpdateParams(dense1)
    optimizer.UpdateParams(dense2)
    optimizer.UpdateParams(dense3)
    optimizer.PostUpdateParams()
end

SaveData('./TrainResults/trainingStats.csv', ConvertTrainingToCSV(epoch_tracker, acc_tracker, loss_tracker))