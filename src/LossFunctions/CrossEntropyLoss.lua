
CrossEntropyLoss = { y_pred = {}, y_true = {} }
require('LossFunctions.BaseLoss')

-- Derived class method new

CrossEntropyLoss.__index = BaseLoss
CrossEntropyLoss.Super = function (self, y_pred, y_true) return BaseLoss:new(self, y_pred, y_true) end

function CrossEntropyLoss:Forward (y_pred, y_true)

   print("forward")
   print(tprint(y_pred))
   print(tprint(y_true))
end

return CrossEntropyLoss

