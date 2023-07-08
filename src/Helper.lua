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