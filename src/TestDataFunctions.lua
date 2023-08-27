local matrix = require('Matrix')
local table = require('Table')

function GenerateBullseye(n)
    --create 2 circles 1 in the other
    -- y^2 + x^2 = 25
    -- y^2 + x^2 = 100

    --first generate random points between -15 and 15
    X = {}
    for i = 1, n do
        local signx, signy
        local randx = math.random()
        local randy = math.random()
        if randx < 0.5 then
        signx = -1
        else
        signx = 1
        end
        if randy < 0.5 then
        signy = -1
        else
        signy = 1
        end
        table.insert(X, {math.random() * 15 * signx, math.random() * 15 * signy})
    end

    --next, classify points
    local y = {}
    for _, v in ipairs(X) do
        local squares = v[2] * v[2] + v[1] * v[1]
        if squares > 10 and squares < 50 then
        table.insert(y, 0)
        elseif squares > 50 and squares < 115 then
        table.insert(y, 1)
        elseif squares > 0 and squares < 10 then
        table.insert(y, 2)
        else
        table.insert(y, 3)
        end
    end

    

    return matrix(X), matrix({y})
end

function GenerateBinaryClasses(n)
    --first generate random points between -15 and 15
    X = {}

    for i = 1, n do
        local signx, signy
        local randx = math.random()
        local randy = math.random()
        if randx < 0.5 then
        signx = -1
        else
        signx = 1
        end

        if randy < 0.5 then
        signy = -1
        else
        signy = 1
        end
        table.insert(X, {math.random() * 15 * signx, math.random() * 15 * signy})
    end

    --next, classify points
    local y = {}
    for _, v in ipairs(X) do
        local squares = v[2] * v[2] + v[1] * v[1]
        if squares < 115 then
        table.insert(y, 1)
        else
        table.insert(y, 0)
        end
    end

    return matrix(X), matrix({y})
end

function Numpy_format(X, y) 

    for _, v in ipairs(X) do
        io.write('[' .. v[1] .. ',' .. v[2] .. '],')
        print()
    end

    for _, v in ipairs(y) do
        io.write(v .. ',')
    end

end

function ConvertPredsToCSV(inputs, output)
    local strData = "inputx, inputy, prediction,\n"
    for i = 1, #output do
        strData = strData ..
        inputs[i][1] .. ',' ..
        inputs[i][2] .. ',' .. 
        output[i] .. ',\n'
    end

    return strData
end

function ConvertRegressPredsToCSV(inputs, output)
    local strData = "inputx, inputy,\n"
    for i = 1, #output do
        strData = strData ..
        inputs[i][1] .. ',' ..
        output[i][1] .. ',\n'
    end

    return strData
end

function ConvertTrainingToCSV(epochs, acc, loss)
    local strData = "epoch, accuracy, loss,\n"
    for i = 1, #epochs do
        strData = strData .. epochs[i] .. ',' ..
        acc[i] .. ',' .. loss[i] .. ',\n'
    end
    return strData
end


function SaveData ( filename, data )
    local file = assert(io.open(filename, "w"))
    file:write(data)
    file:close()
 end 

 function SinData (start, stop, step)
    --create a sine wave as our training data
    local X_train = table.arrange(start, stop, step)

    local y_train = {} --Input training coords
    local c = 0
    for _, v in ipairs(X_train) do
    table.insert(y_train, math.sin(v * math.pi * 2))
    end

    X_train = matrix.transpose(matrix({X_train}))
    y_train = matrix({y_train})

    return X_train, y_train
 end
