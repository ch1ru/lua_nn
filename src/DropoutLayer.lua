require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')

local matrix = require 'Matrix'
local table = require "Table"

DropoutLayer = {}

function DropoutLayer:new(rate)
    local o = {}
    setmetatable(o, {__index = self})
    o.name = "dropout_layer"
    o.rate = 1- rate
    --functions
    o.forward = function (inputs, training) return self:Forward(o, inputs, training) end
    o.backward = function (dvalues) return self:Backward(o, dvalues) end
    return o
end

function DropoutLayer:Forward(self, inputs, training) 
    self.inputs = inputs

    if not training then
        self.outputs = matrix.copy(inputs)
        return
    end

    --ADD random binomial

    self.output = inputs * self.binary_mask
end

function DropoutLayer:Backward(self, dvalues)
    self.dinputs = dvalues * self.binary_mask
end

return DropoutLayer
