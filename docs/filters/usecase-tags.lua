--- Append the published studies that exercise the feature a page documents.
---
--- The list comes from `_static/usecase_tags.lua`, derived by `scripts/build_usecase_tags.py` from the specification slots the use-case corpus actually writes. Nothing here is typed by hand, so a page cannot claim a study that stopped using its feature. A page may suppress the block with `usecase-tags: false`.

local tags = nil

local function project_root()
  return (quarto and quarto.project and quarto.project.directory) or "."
end

local function page_key()
  local root, file = project_root(), quarto and quarto.doc and quarto.doc.input_file
  if not file then return nil end
  if file:sub(1, #root + 1) == root .. "/" then return file:sub(#root + 2) end
  return file
end

local function escape(text)
  return (text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

--- The corpus keys studies as author-surnames plus year, so `TsodyksMarkram1997` reads back as `Tsodyks & Markram (1997)`.
local function citation(key, year)
  local names = key:gsub("%d%d%d%d$", ""):gsub("(%l)(%u)", "%1 & %2")
  if year ~= "" then return names .. " (" .. year .. ")" end
  return names
end

local function item(entry)
  local label = escape(citation(entry.key, entry.year))
  local title = escape(entry.title)
  if entry.doi ~= "" then
    return string.format('<li><a href="https://doi.org/%s"><strong>%s</strong></a> — %s</li>',
      escape(entry.doi), label, title)
  end
  return string.format('<li><strong>%s</strong> — %s</li>', label, title)
end

function Pandoc(doc)
  if doc.meta["usecase-tags"] == false then return doc end
  if not tags then
    local ok, loaded = pcall(dofile, project_root() .. "/_static/usecase_tags.lua")
    tags = (ok and loaded) or {}
  end

  local key = page_key()
  local entries = key and tags[key] or nil
  if not entries or #entries == 0 then return doc end

  local items = {}
  for _, entry in ipairs(entries) do items[#items + 1] = item(entry) end

  local html = string.format(
    '<aside class="tvbo-usecases"><h2 class="tvbo-usecases__head">' ..
    '<i class="fa-solid fa-flask-vial" aria-hidden="true"></i>Used in published replications</h2>' ..
    '<p class="tvbo-usecases__lede">%d peer-reviewed %s whose TVB-O recipe writes what this page documents.</p>' ..
    '<ul>%s</ul></aside>',
    #entries, #entries == 1 and "study" or "studies", table.concat(items, ""))

  table.insert(doc.blocks, pandoc.RawBlock("html", html))
  return doc
end
