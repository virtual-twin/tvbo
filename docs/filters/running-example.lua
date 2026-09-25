--- Link each ② SPECIFY page back to the stage of the running example that grows its block.
---
--- Both ends come from `_static/running_example.yml`, so a page cannot point at a stage that no longer exists. The running-example page itself is skipped, and any page may opt out with `running-example: false`.

local index = nil

local function project_root()
  return (quarto and quarto.project and quarto.project.directory) or "."
end

local function page_key()
  local root, file = project_root(), quarto and quarto.doc and quarto.doc.input_file
  if not file then return nil end
  if file:sub(1, #root + 1) == root .. "/" then return file:sub(#root + 2) end
  return file
end

--- Depth-relative href, since Quarto resolves a raw-HTML link against the page, not the project.
local function relative_to(from, target)
  local up = ""
  for _ in from:gmatch("/") do up = up .. "../" end
  return up .. (target:gsub("%.qmd$", ".html"))
end

--- A deliberately small YAML reader: this file's shape is fixed by the schema above, so a full parser would be dead weight.
local function parse()
  if index then return index end
  index = {}
  local handle = io.open(project_root() .. "/_static/running_example.yml", "r")
  if not handle then return index end

  local target_page, stage_id, stage_number, stage_what
  for line in handle:lines() do
    local page = line:match("^page:%s*(%S+)")
    if page then target_page = page end
    local id = line:match("^%s*%-%s*id:%s*(%S+)")
    if id then stage_id, stage_number, stage_what = id, nil, nil end
    local number = line:match("^%s*number:%s*(%d+)")
    if number then stage_number = tonumber(number) end
    local what = line:match('^%s*what:%s*"(.*)"%s*$')
    if what then stage_what = what end
    local entry = line:match("^%s*%-%s+([%w%./_%-]+%.qmd)%s*$")
    if entry and stage_id then
      index[entry] = {id = stage_id, number = stage_number, what = stage_what, page = target_page}
    end
  end
  handle:close()
  return index
end

function Pandoc(doc)
  if doc.meta["running-example"] == false then return doc end

  local key = page_key()
  if not key then return doc end

  local entry = parse()[key]
  if not entry then return doc end
  if key == entry.page then return doc end
  -- The reader above is deliberately small, so a stage whose `what:` is unquoted or which omits `number:` comes back with those fields nil. Reporting that is the filter's job; indexing them anyway would abort the render of a page that is otherwise fine.
  if not (entry.what and entry.number and entry.page) then
    quarto.log.warning("running-example: stage " .. tostring(entry.id) ..
      " is missing a quoted `what:` or a `number:` in _static/running_example.yml")
    return doc
  end

  local href = relative_to(key, entry.page) .. "#" .. entry.id
  local what = entry.what:gsub("`([^`]+)`", "<code>%1</code>")
  local html = string.format(
    '<p class="tvbo-thread"><i class="fa-solid fa-diagram-next" aria-hidden="true"></i>' ..
    'Part of <a href="%s">the running example</a>, where stage %d adds %s.</p>',
    href, entry.number, what)

  local at = (#doc.blocks > 0 and doc.blocks[1].t == "RawBlock") and 2 or 1
  table.insert(doc.blocks, at, pandoc.RawBlock("html", html))
  return doc
end
