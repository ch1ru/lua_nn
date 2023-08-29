require('Helper')
require('Relu')
require('BaseLoss')
require('CrossEntropyLoss')

local matrix = require 'Matrix'
local table = require "Table"

InputLayer = {}

function InputLayer:new ()
   local o = {}
   setmetatable(o, {__index = self})
   o.name = "input_layer"
   --class functions
   o.forward = function (inputs) return self:Forward(o, inputs) end
   o.backward = function (output, y) return self:Backward(o, output, y) end
   return o
end

function InputLayer:Forward(self, inputs)
   self.output = inputs
end

return InputLayer