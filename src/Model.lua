require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')
require('InputLayer')

local matrix = require 'Matrix'
local table = require "Table"

Model = {}

function Model:new ()
   local o = {}
   setmetatable(o, {__index = self})
   o.layers = {}
   --class functions
   o.add = function (layer) return self:Add(o, layer) end
   o.set = function (loss, optimizer, accuracy) return self:Set(o, loss, optimizer, accuracy) end
   o.finalize = function () return self:Finalize(o) end
   o.train = function (X, y, epochs, printEvery, validationData) return self:Train(o, X, y, epochs, printEvery, validationData) end
   o.forward = function (X, training) return self:Forward(o, X, training) end
   o.backward = function (output, y) return self:Backward(o, output, y) end
   return o
end

function Model:Add(self, layer)
    table.insert(self.layers, layer)
end

function Model:Set(self, loss, optimizer, accuracy)
    self.loss = loss
    self.optimizer = optimizer
    self.accuracy = accuracy
end

function Model:Finalize(self)
    self.inputLayer = InputLayer:new()

    local layerCount = #self.layers

    self.trainableLayers = {}

    for i = 1, layerCount do
        if i == 1 then
            self.layers[i].prev = self.inputLayer
            self.layers[i].next = self.layers[i+1]
        elseif i < layerCount then
            self.layers[i].prev = self.layers[i-1]
            self.layers[i].next = self.layers[i+1]
        else
            self.layers[i].prev = self.layers[i-1]
            self.layers[i].next = self.loss
            self.output_layer_activation = self.layers[i]
        end

        if self.layers[i].weights ~= nil then
            table.insert(self.trainableLayers, self.layers[i])
        end
    end

    self.loss.rememberTrainableLayers(self.trainableLayers)
end

function Model:Train(self, X, y, epochs, printEvery, validationData)
    
    self.accuracy.init(y)
    
    for epoch = 1, epochs do
        local output = self.forward(X)

        local data_loss, regularization_loss = self.loss.calculate(output, y)
        local loss = data_loss + regularization_loss

        local preds = self.output_layer_activation.predictions(output)
        local acc = self.accuracy.calculate(preds, y)

        self.backward(output, y)

        self.optimizer.PreUpdateParams()
        for i = 1, #self.trainableLayers do
            local layer = self.trainableLayers[i]
            self.optimizer.UpdateParams(layer)
        end
        self.optimizer.PostUpdateParams()

        --output summary
        if epoch % printEvery == 0 then
            acc = 0.0
            print(string.format("Epoch: %d, Acc: %f, Loss: %f, learning rate: %f", epoch, acc, loss, self.optimizer.currentlr))
            --SaveData('./TrainResults/output.csv', ConvertRegressPredsToCSV(matrix(X), matrix(y)))
        end
    end
end

function Model:Forward(self, X, training)
    self.inputLayer.forward(X)
    local x_out
    for i = 1, #self.layers do
        local layer = self.layers[i]
        layer.forward(layer.prev.output)
        x_out = layer.output
    end
    return x_out
end

function Model:Backward(self, output, y)
    self.loss.backward(output, matrix.transpose(y))

    for i = #self.layers, 1, -1 do
        local layer = self.layers[i]
        layer.backward(layer.next.dinputs)
    end
end

return Model