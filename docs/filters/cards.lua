--- Render an overview page's index of pages as a card grid instead of a table.
---
--- An overview page lists the pages beneath it as a pipe table wrapped in `::: {.cards}`: a link in the first column, a sentence in the second, and an optional cover in the third. A cover is either an image, which becomes the card's thumbnail, or a Font Awesome class such as `fa-solid fa-plug`, which becomes a tile drawn in the target page's phase colour. A row that names neither falls back to the phase's own icon, so the grid always has covers. Colours and icons come from `_static/phase_map.lua` and never from this filter. Keeping the source a table is what makes the list editable; the cards are the presentation.

local map = nil

local function project_root()
  return (quarto and quarto.project and quarto.project.directory) or "."
end

local function load_map()
  if map then return end
  local ok, loaded = pcall(dofile, project_root() .. "/_static/phase_map.lua")
  if not ok then
    quarto.log.warning("cards: no _static/phase_map.lua; run scripts/build_phase_map.py (the pre-render does) to give cards their icons")
  end
  map = (ok and loaded) or {}
end

--- The project-relative directory of the page being rendered, which every href in it is relative to.
local function page_dir()
  local root, file = project_root(), quarto and quarto.doc and quarto.doc.input_file
  if not file then return "" end
  local rel = file
  if file:sub(1, #root + 1) == root .. "/" then rel = file:sub(#root + 2) end
  return rel:match("^(.*)/[^/]*$") or ""
end

--- A page's key in the phase map: its href resolved against the current page and collapsed to a project-relative path.
local function resolve(href, dir)
  local parts = {}
  for segment in ((dir ~= "" and dir .. "/" or "") .. href):gmatch("[^/]+") do
    if segment == ".." then
      table.remove(parts)
    elseif segment ~= "." then
      table.insert(parts, segment)
    end
  end
  return table.concat(parts, "/")
end

local function to_html_href(href)
  return (href:gsub("%.qmd$", ".html"):gsub("%.md$", ".html"))
end

local function escape(text)
  return (text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;"))
end

local function inlines_to_html(inlines, block)
  return pandoc.write(pandoc.Pandoc({ (block or pandoc.Plain)(inlines) }), "html")
end

--- The first link in a cell, which is the card's destination and title.
local function first_link(blocks)
  local found = nil
  pandoc.walk_block(pandoc.Div(blocks), {
    Link = function(link)
      if not found then found = link end
    end,
  })
  return found
end

--- A row's optional third column, read as an image source or as a Font Awesome class.
local function cover(cells)
  if #cells < 3 then return nil, nil end
  local blocks = cells[3].contents
  local src = nil
  pandoc.walk_block(pandoc.Div(blocks), {
    Image = function(image)
      if not src then src = image.src end
    end,
  })
  if src then return src, nil end
  local text = pandoc.utils.stringify(pandoc.Div(blocks)):gsub("^%s+", ""):gsub("%s+$", "")
  if text == "" then return nil, nil end
  if text:match("^fa%-") then return nil, text end
  return text, nil
end

local function thumb(src, icon, entry)
  if src then
    return string.format('<span class="tvbo-card__thumb"><img src="%s" alt=""></span>', escape(src))
  end
  return string.format('<span class="tvbo-card__thumb tvbo-card__thumb--tile"><i class="%s" aria-hidden="true"></i></span>',
    escape(icon or (entry and entry.icon) or "fa-solid fa-file-lines"))
end

local function card(row, dir)
  local cells = row.cells
  local link = first_link(cells[1].contents)
  if not link then return nil end
  local entry = map[resolve(link.target, dir)]
  local description = #cells > 1 and inlines_to_html(pandoc.utils.blocks_to_inlines(cells[2].contents), pandoc.Para) or ""
  local src, icon = cover(cells)
  return string.format(
    '<a class="tvbo-card" href="%s" style="--tile: %s">%s<span class="tvbo-card__text"><span class="tvbo-card__title">%s</span>%s</span></a>',
    escape(to_html_href(link.target)), (entry and entry.color) or "#5c6f6e",
    thumb(src, icon, entry), inlines_to_html(link.content), description)
end

function Div(div)
  if not div.classes:includes("cards") then return nil end
  load_map()
  local dir, cards = page_dir(), {}
  pandoc.walk_block(div, {
    Table = function(table_)
      for _, body in ipairs(table_.bodies) do
        for _, row in ipairs(body.body) do
          local rendered = card(row, dir)
          if rendered then table.insert(cards, rendered) end
        end
      end
    end,
  })
  if #cards == 0 then return nil end
  return pandoc.RawBlock("html", '<div class="tvbo-cards">' .. table.concat(cards) .. "</div>")
end
