LinearActivation = {}

function LinearActivation:new()
    local o = {}
    setmetatable(o, self)
    o.forward = function (x) return self:Forward(o, x) end
    o.backward = function (dvalues) return self:Backward(o, dvalues) end
    return o
end

function LinearActivation:Forward(self, inputs)
    self.inputs = inputs
    self.output = inputs
end

function LinearActivation:Backward(self, dvalues)
    self.dinputs = dvalues
end

return LinearActivation