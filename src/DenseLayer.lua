require('Helper')
require('ActivationFunctions.Relu')
require('LossFunctions.BaseLoss')
require('LossFunctions.CrossEntropyLoss')
require('Tensor')

local matrix = require 'Matrix'
local table = require "LuaTable"


DenseLayer = {n_inputs = 0, n_neurons = 0, weights = Tensor:new(nil, {}), bias = Tensor:new(nil, {}) }

-- Derived class method new

function DenseLayer:new (o, n_inputs, n_neurons, weights, bias)
   local o = o or {}
   setmetatable(o, self)
   self.__index = self
   o.n_inputs = n_inputs
   o.n_neurons = n_neurons
   o.weights = weights or Tensor:new(nil, table.normal({n_inputs, n_neurons}))
   o.bias = bias or Tensor:Zeros({1, n_neurons})
   
   return o
end

function DenseLayer:Forward(inputs)
   --apply relu activation
   local inputs = ReLU(inputs)
   self.output = (inputs * self.weights) + self.bias
end

local loss = CrossEntropyLoss:Super({1}, {4})
loss:Calculate()

local t = { 1, 7, 2, 3, -4, 5 }

-- remove negative values
--t = table.accept(t, table.positive)

-- create an enhanched table from an old one
local tb = table(t)

print(tb.min(tb))

local m1 = matrix{{tb,tb,tb},{tb,tb,tb}}
local m2 = matrix{{-8,1,3},{5,2,1}}
print(m1)