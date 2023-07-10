require('Math')

function tprint (tbl, index)
    local indent = 0
    local toprint = string.rep(" ", indent) .. "{\r\n"
    indent = indent + 2 
    for k, v in pairs(tbl) do
      toprint = toprint .. string.rep(" ", indent)
      if (type(k) == "number" and index == true) then
        toprint = toprint .. "[" .. k .. "] = "
      elseif (type(k) == "string" and index == true) then
        toprint = toprint  .. k ..  "= "   
      end
      if (type(v) == "number") then
        toprint = toprint .. v .. ",\r\n"
      elseif (type(v) == "string") then
        toprint = toprint .. "\"" .. v .. "\",\r\n"
      elseif (type(v) == "table") then
        toprint = toprint .. tprint(v, indent + 2) .. ",\r\n"
      else
        toprint = toprint .. "\"" .. tostring(v) .. "\",\r\n"
      end
    end
    toprint = toprint .. string.rep(" ", indent-2) .. "}"
    return toprint
end

function table.explode(tbl, groups)
  local pointer = 1
  local pointerEnd = #tbl
  local finalTable = {}
  while pointer <= pointerEnd do
    local currTable = {}
    for i = 1, groups do
      table.insert(currTable, tbl[pointer])
      pointer = pointer + 1
    end
    table.insert(finalTable, currTable)
  end
  return finalTable
end


function table.normal(size, mean, variance, sf) 
  if mean == nil then mean = 0 end
  if variance == nil then variance = 1 end
  if sf == nil then sf = 1 end
  local t = {}
  local n = 1
  --calculate total size
  for _, v in ipairs(size) do
    n = n * v
  end

  for i = 1, n do
    local x = Gaussian(mean, variance)
    table.insert(t, x * sf)
  end

  return ResizeTable(t, size)
end

--Converts a tensor of any dimension into a single dimension
function ConvertTo1D(tensor)
  local newTable = {}
  for _, v in ipairs(tensor) do
      if type(v) == 'table' then
          local v = ConvertTo1D(v)
          for _, w in ipairs(v) do
              table.insert(newTable, w)
          end
      else
          table.insert(newTable, v)
      end
  end
  return newTable
end

--Create multi-dimensional table from a single table
function ResizeTable(singleTable, newShape)
  local newTable = ConvertTo1D(singleTable)
  
  for i = #newShape, 2, -1 do
      newTable = table.explode(newTable, newShape[i])
  end
  return newTable
end

function Tensor:Exp(x)
  local shape = x:Shape()
  local t = {}
  local singleTensor = ConvertTo1D(x.tensor)
  for _, v in ipairs(singleTensor) do
      table.insert(t, math.exp(v))
  end
  --convert back to original shape
  return Tensor:new(nil, ResizeTable(t, shape))
end

