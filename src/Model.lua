require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')

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
    self.opttimizer = optimizer
    self.accuracy = accuracy
end

function Model:Finalize(self)

end

function Model:Train(self, X, y, epochs, printEvery, validationData)

end

function Model:Forward(self, X, training)
    self.inputLayer.forward(X)
    local x_out = self.inputLayer.output
    for layer in self.layers do
        layer.forward(layer.prev.output)
        x_out = layer.output
    end
    return x_out
end

function Model:Backward(self, output, y)

end

return Model