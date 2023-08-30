local math = require('Math')
local table = require('Table')
local matrix = require('Matrix')

DataLoader = { X = {}, y = {}, batch_size = nil }

function DataLoader:new(X, y, batch_size, shuffle)
    local o = {}
    setmetatable(o, self)
    self.__index = self
    o.name = "dataloader"
    o.batch_size = batch_size or y:columns()
    o.shuffle = shuffle

    if shuffle then
        
    end

    if o.batch_size < y:columns() then
        local batches_X = {}
        local batch_X = matrix:new(o.batch_size, X:columns())
        local batches_y = {}
        local batch_y = matrix:new(y:rows(), o.batch_size)
        local count = 1

        for i = 1, y:columns() do

            batch_X[count] = X[i]
            batch_y[1][count] = y[1][i]

            
        
            if count >= o.batch_size then
                table.insert(batches_X, batch_X)
                table.insert(batches_y, batch_y)
                batch_X = matrix:new(o.batch_size, X:columns())
                batch_y = matrix:new(y:rows(), o.batch_size)
                count = 1
            end

            count = count + 1

        end
        
        o.X = batches_X
        o.y = batches_y
        
    else
        o.X = {X}
        o.y = {y}
    end

    --class functions
    o.get_batch = function () return self:get_batch(o) end
    return o
end

function DataLoader:get_batch(self)
    return self.X, self.y --table of batches
end

return DataLoader