require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')
require('InputLayer')
require('SoftmaxCrossEntropyLoss')
local Dataloader = require('DataLoader')

local matrix = require 'Matrix'
local table = require "Table"

Model = {}

function Model:new ()
   local o = {}
   setmetatable(o, {__index = self})
   o.name = "model"
   o.layers = {}
   --class functions
   o.add = function (layer) return self:Add(o, layer) end
   o.set = function (loss, optimizer, accuracy) return self:Set(o, loss, optimizer, accuracy) end
   o.finalize = function () return self:Finalize(o) end
   o.train = function (dataloader, epochs, printEvery) return self:Train(o, dataloader, epochs, printEvery) end
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

    if self.layers[#self.layers].name == "softmax"
        and self.loss.name == "crossentropy_loss" then
            self.softmax_classifier_output = SoftmaxCrossEntropyLoss:new()
    end
end

function Model:Train(self, dataloader, epochs, printEvery)
    
    for epoch = 1, epochs do

        local X_batches, y_batches = dataloader.get_batch()

        local accuracy_epoch = {}
        local loss_epoch = {}

        --cycle through batches
        for i = 1, #y_batches do 

                local X = X_batches[i]
                local y = y_batches[i]

                self.accuracy.init(y)

                local output = self.forward(X)

                local data_loss, regularization_loss = self.loss.calculate(output, y)
                local loss = data_loss + regularization_loss
                table.insert(loss_epoch, loss)

                local preds = self.output_layer_activation.predictions(output)

                local acc = self.accuracy.calculate(preds, y)
                table.insert(accuracy_epoch, acc)

                self.backward(output, y)

                self.optimizer.PreUpdateParams()
                for i = 1, #self.trainableLayers do
                    local layer = self.trainableLayers[i]
                    self.optimizer.UpdateParams(layer)
                end

                self.optimizer.PostUpdateParams()
              
                SaveData('./TrainResults/output.csv', ConvertPredsToCSV(matrix(X), preds))

                
        end

        local loss = table.avg(loss_epoch)
        local acc = table.avg(accuracy_epoch)

        --output summary
        if epoch % printEvery == 0 then
            print(string.format("Epoch: %d, Acc: %f, Loss: %f, learning rate: %f", epoch, acc, loss, self.optimizer.currentlr))
            
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

    if self.softmax_classifier_output ~= nil then
        --calls backward on condensed softmax/cross entropy function

        self.softmax_classifier_output.backward(output, y)

        self.layers[#self.layers].dinputs = self.softmax_classifier_output.dinputs

        --backpropogate through all layers but last
        for i = #self.layers - 1, 1, -1 do
            local layer = self.layers[i]
            layer.backward(layer.next.dinputs)
        end

        return
    end

    self.loss.backward(output, y)

    --backpropogate through all layers
    for i = #self.layers, 1, -1 do
        local layer = self.layers[i]
        layer.backward(layer.next.dinputs)
    end
end

return Model