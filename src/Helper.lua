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

function table.slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
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

function table.max(a)
  local values = {}

  for k,v in pairs(a) do
    values[#values+1] = v
  end
  table.sort(values) -- automatically sorts lowest to highest

  return values[#values]
end

function table.min(a)
  local values = {}

  for k,v in pairs(a) do
    values[#values+1] = v
  end
  table.sort(values) -- automatically sorts lowest to highest

  return values[1]
end


function table.normal(size, mean, variance) 
  if mean == nil then mean = 0 end
  if variance == nil then variance = 1 end
  local t = {}
  local n = 1
  --calculate total size
  for _, v in ipairs(size) do
    n = n * v
  end

  for i = 1, n do
    local x = Gaussian(mean, variance)
    table.insert(t, x)
  end

  return ResizeTable(t, size)
end

function table.ones(size) 
  local t = {}
  local n = 1
  --calculate total size
  for _, v in ipairs(size) do
    n = n * v
  end

  --fill 1D array with zeros
  for i = 1, n do 
    table.insert(t, 1)
  end

  --resize array
  return ResizeTable(t, size)
end

function table.zeros(size) 
  local t = {}
  local n = 1
  --calculate total size
  for _, v in ipairs(size) do
    n = n * v
  end

  --fill 1D array with zeros
  for i = 1, n do 
    table.insert(t, 0)
  end

  --resize array
  return ResizeTable(t, size)
end

function table.__mul(t, scalar)

  if type(t) == 'number' and type(scalar) == 'table' then
     local tmp = t
     t = scalar
     scalar = tmp
  end

  local newTable = {}
  for _, v in ipairs(t) do
     table.insert(newTable, v * scalar)
  end
  return newTable
end

