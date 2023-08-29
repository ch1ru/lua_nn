local matrix = require('Matrix')
local table = require('Table')

Optimizer = {}

---------------Stochastic Gradient Descent-----------------

Optimizer.SGD = function (learningRate, decay, momentum) return SGD:new(learningRate, decay, momentum) end

SGD = {}

function SGD:new (learningRate, decay, momentum)
   local o = {}
   setmetatable(o, {__index = self})
   o.name = "optim_sgd"
   o.lr = learningRate or 0.01
   o.currentlr = o.lr
   o.decay = decay or 0
   o.iters = 0
   o.momentum = momentum or 0
   --functions
   o.UpdateParams = function (layer) return self:UpdateParams(o, layer) end
   o.PreUpdateParams = function () return self:PreUpdateParams(o) end
   o.PostUpdateParams = function () return self:PostUpdateParams(o) end
   return o
end

function SGD:PreUpdateParams(self)
    if self.decay then
        self.currentlr = self.lr * (1 / (1 + self.decay * self.iters))
    end
end

function SGD:UpdateParams(self, layer)
    local weightsUpdates, biasUpdates
    if self.momentum then
        if layer.weightMomentums == nil then
            layer.weightMomentums = matrix:new(layer.weights:rows(), layer.weights:columns(), 0)
            layer.biasMomentums = matrix:new(layer.biases:rows(), layer.biases:columns(), 0)
        end

        --build weights updates
        weightsUpdates = self.momentum * layer.weightMomentums - self.currentlr * layer.dweights
        layer.weightMomentums = weightsUpdates

        --build bias updates
        biasUpdates = self.momentum * layer.biasMomentums - self.currentlr * layer.dbiases
        layer.biasMomentums = biasUpdates
    else
        weightsUpdates = -self.currentlr * layer.dweights
        biasUpdates = matrix.mulnum(layer.dbiases, -self.currentlr)
    end

    layer.weights = layer.weights + weightsUpdates
    layer.biases = layer.biases + biasUpdates
    
end

function SGD:PostUpdateParams(self)
    self.iters = self.iters + 1
end

------------Adagrad----------------

Optimizer.Adagrad = function (learningRate, decay, epsilon) return Adagrad:new(learningRate, decay, epsilon) end

Adagrad = {}

function Adagrad:new (learningRate, decay, epsilon)
   local o = {}
   setmetatable(o, {__index = self})
   o.name = "optim_adagrad"
   o.lr = learningRate or 0.01
   o.currentlr = o.lr
   o.decay = decay or 0
   o.iters = 0
   o.epsilon = epsilon or 0
   --functions
   o.UpdateParams = function (layer) return self:UpdateParams(o, layer) end
   o.PreUpdateParams = function () return self:PreUpdateParams(o) end
   o.PostUpdateParams = function () return self:PostUpdateParams(o) end
   return o
end

function Adagrad:PreUpdateParams(self)
    if self.decay then
        self.currentlr = self.lr * (1 / (1 + self.decay * self.iters))
    end
end

function Adagrad:UpdateParams(self, layer)
    local weightsUpdates, biasUpdates

    if layer.weightCache == nil then
        layer.weightCache = matrix:new(layer.weights:rows(), layer.weights:columns(), 0)
        layer.biasCache = matrix:new(layer.biases:rows(), layer.biases:columns(), 0)
    end

     
    layer.weightCache = layer.weightCache + layer.dweights * layer.dweights
    layer.biasCache = layer.biasCache + layer.dbiases * layer.dbiases

    layer.weights = layer.weights + (matrix.mulnum(layer.dweights, -self.currentlr) / (matrix.sqrt(layer.weightCache) + self.epsilon))
    layer.biases = layer.biases + (matrix.mulnum(layer.dbiases, -self.currentlr) / (matrix.sqrt(layer.biasCache) + self.epsilon))
    
end

function Adagrad:PostUpdateParams(self)
    self.iters = self.iters + 1
end

-----------------RMSPROP--------------------

Optimizer.RMSProp = function (learningRate, decay, epsilon) return RMSProp:new(learningRate, decay, epsilon) end

RMSProp = {}

function RMSProp:new (learningRate, decay, epsilon, rho)
   local o = {}
   setmetatable(o, {__index = self})
   o.name = "optim_rmsprop"
   o.lr = learningRate or 0.01
   o.currentlr = o.lr
   o.decay = decay or 0
   o.iters = 0
   o.epsilon = epsilon or 0
   o.rho = rho
   --functions
   o.UpdateParams = function (layer) return self:UpdateParams(o, layer) end
   o.PreUpdateParams = function () return self:PreUpdateParams(o) end
   o.PostUpdateParams = function () return self:PostUpdateParams(o) end
   return o
end

function RMSProp:PreUpdateParams(self)
    if self.decay then
        self.currentlr = self.lr * (1 / (1 + self.decay * self.iters))
    end
end

function RMSProp:UpdateParams(self, layer)
    local weightsUpdates, biasUpdates

    if layer.weightCache == nil then
        layer.weightCache = matrix:new(layer.weights:rows(), layer.weights:columns(), 0)
        layer.biasCache = matrix:new(layer.biases:rows(), layer.biases:columns(), 0)
    end

     
    layer.weightCache = matrix.mulnum(layer.weightCache, self.rho) + matrix.mulnum(layer.dweights * layer.dweights, (1 - self.rho))
    layer.biasCache = matrix.mulnum(layer.biasCache, self.rho) + matrix.mulnum(layer.dbiases * layer.dbiases, (1 - self.rho))
    
    layer.weights = layer.weights + (matrix.mulnum(layer.dweights, -self.currentlr) / (matrix.sqrt(layer.weightCache) + self.epsilon))
    layer.biases = layer.biases + (matrix.mulnum(layer.dbiases, -self.currentlr) / (matrix.sqrt(layer.biasCache) + self.epsilon))
    
end

function RMSProp:PostUpdateParams(self)
    self.iters = self.iters + 1
end

------------------Adam-------------------

Optimizer.Adam = function (learningRate, decay, epsilon) return Adam:new(learningRate, decay, epsilon) end

Adam = {}

function Adam:new (learningRate, decay, epsilon, beta1, beta2)
   local o = {}
   setmetatable(o, {__index = self})
   o.name = "optim_adam"
   o.lr = learningRate or 0.01
   o.currentlr = o.lr
   o.decay = decay or 0
   o.iters = 0
   o.epsilon = epsilon or 1e-7
   o.beta1 = beta1 or 0.9
   o.beta2 = beta2 or 0.999
   --functions
   o.UpdateParams = function (layer) return self:UpdateParams(o, layer) end
   o.PreUpdateParams = function () return self:PreUpdateParams(o) end
   o.PostUpdateParams = function () return self:PostUpdateParams(o) end
   return o
end

function Adam:PreUpdateParams(self)
    if self.decay then
        self.currentlr = self.lr * (1 / (1 + self.decay * self.iters))
    end
end

function Adam:UpdateParams(self, layer)

    if layer.weightCache == nil then
        layer.weightMomentums = matrix:new(layer.weights:rows(), layer.weights:columns(), 0)
        layer.biasMomentums = matrix:new(layer.biases:rows(), layer.biases:columns(), 0)
        layer.weightCache = matrix:new(layer.weights:rows(), layer.weights:columns(), 0)
        layer.biasCache = matrix:new(layer.biases:rows(), layer.biases:columns(), 0)
    end

    layer.weightMomentums = matrix.mulnum(layer.weightMomentums, self.beta1) + matrix.mulnum(layer.dweights, (1 - self.beta1))
    layer.biasMomentums = matrix.mulnum(layer.biasMomentums, self.beta1) + matrix.mulnum(layer.dbiases, (1 - self.beta1))

    local weight_momentums_corrected = matrix.divnum(layer.weightMomentums, 1 - self.beta1 ^ (self.iters + 1))
    local bias_momentums_corrected = matrix.divnum(layer.biasMomentums, 1 - self.beta1 ^ (self.iters + 1))

    layer.weightCache = matrix.mulnum(layer.weightCache, self.beta2) + matrix.mulnum(matrix.powNum(layer.dweights, 2), 1 - self.beta2)
    layer.biasCache = matrix.mulnum(layer.biasCache, self.beta2) + matrix.mulnum(matrix.powNum(layer.dbiases, 2), 1 - self.beta2)

    local weight_cache_corrected = matrix.divnum(layer.weightCache, 1 - self.beta2 ^ (self.iters + 1))
    local bias_cache_corrected = matrix.divnum(layer.biasCache, 1 - self.beta2 ^ (self.iters + 1))

    layer.weights = layer.weights + matrix.divide((matrix.mulnum(weight_momentums_corrected, -self.currentlr)), 
        matrix.addNum(matrix.powNum(weight_cache_corrected, 0.5), self.epsilon))
    layer.biases = layer.biases + matrix.divide((matrix.mulnum(bias_momentums_corrected, -self.currentlr)),
        matrix.addNum(matrix.powNum(bias_cache_corrected, 0.5), self.epsilon))


end


function Adam:PostUpdateParams(self)
    self.iters = self.iters + 1
end


return Optimizer