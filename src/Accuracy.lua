local matrix = require('Matrix')
require('Table')

function CalculateAcc(preds, y_true)
    
    local pred_t = {}

    for i = 1, #preds do
        for j = 1, #preds[i] do
            if table.max(preds[i])[1] == preds[i][j] then
                table.insert(pred_t, j-1)
            end
        end
    end

    local correct_counter = 0
    for i = 1, #y_true[1] do
        if y_true[1][i] == pred_t[i] then
            correct_counter = correct_counter + 1
        end
    end

    local mean = correct_counter / #y_true[1]

    return mean
    
end