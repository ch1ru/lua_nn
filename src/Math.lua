--Probability Distribution Functions from Rosetta code
--source: https://rosettacode.org/wiki/Statistics/Normal_distribution#Lua

function Gaussian (mean, variance)
    return  math.sqrt(-2 * variance * math.log(math.random())) *
            math.cos(2 * math.pi * math.random()) + mean
end
 
function Mean (t)
    local sum = 0
    for k, v in pairs(t) do
        sum = sum + v
    end
    return sum / #t
end
 
function Std (t)
    local squares, avg = 0, Mean(t)
    for k, v in pairs(t) do
        squares = squares + ((avg - v) ^ 2)
    end
    local variance = squares / #t
    return math.sqrt(variance)
end
 
function ShowHistogram (t) 
    local lo = math.ceil(math.min(table.unpack(t)))
    local hi = math.floor(math.max(table.unpack(t)))
    local hist, barScale = {}, 200
    for i = lo, hi do
        hist[i] = 0
        for k, v in pairs(t) do
            if math.ceil(v - 0.5) == i then
                hist[i] = hist[i] + 1
            end
        end
        io.write(i .. "\t" .. string.rep('=', hist[i] / #t * barScale))
        print(" " .. hist[i])
    end
end

