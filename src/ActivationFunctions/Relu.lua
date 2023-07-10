--ReLU (Rectified Linear Activation Function) returns the input when positive and 0 when negative
-- y = {x x>0, 0 x<=0}

function ReLU(x)
    local t = {}
    for _, v in ipairs(x) do
        table.insert(t, math.max(v, 0))
    end
    return Tensor:new(nil, t)
end