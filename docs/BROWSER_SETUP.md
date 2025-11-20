# TVB-O Browser Setup

## Overview

The TVB-O Browser is fully integrated into the Quarto documentation website. It provides an interactive interface to search and explore TVB-O models, atlases, studies, networks, couplings, and schemas.

## Architecture

### Source Files (docs/browser/)
- `browser_template.js` - Full-featured browser implementation with faceted search
- `model_builder.js` - Interactive model builder UI
- `styles.css` - Browser styling
- `builder.css` - Model builder styling
- `builder.html` - Model builder modal template
- `template.html` - Standalone browser page
- `img/` - Static images (favicon, icons)

### Generated Files (docs/_site/browser/)
All files are automatically generated during `quarto render`:

- `browser.js` - Copied from `browser_template.js`
- `data.js` - Generated from YAML files in `database/`
- `schema.js` - Inferred schema for faceted search
- `index.html` - Standalone browser (from template.html)
- All CSS, HTML, and asset files copied from source

### Integration Points

1. **browser.qmd** - Quarto page that embeds the browser in the website frame
   - Includes all necessary HTML structure
   - Loads browser scripts and data
   - Provides Model Builder button
   - Maintains site header/footer

2. **generate_browser.py** - Post-render script that:
   - Copies all source files from `docs/browser/` to `docs/_site/browser/`
   - Loads YAML data from `database/` directories
   - Generates `data.js` with all items
   - Generates `schema.js` for search configuration
   - Optionally generates reports and thumbnails (if tvbo/matplotlib available)
   - Fixes browser.html to add required elements

## Clean Build Workflow

To perform a clean build and ensure all files are correctly placed:

```bash
cd /path/to/tvbo
source .venv/bin/activate
cd docs

# Remove old build
rm -rf _site

# Render everything
quarto render
```

The post-render script (`scripts/generate_browser.py`) automatically:
1. Creates `_site/browser/` directory structure
2. Copies all source files from `docs/browser/`
3. Generates `data.js` from database YAML files
4. Generates `schema.js` for search configuration
5. Ensures browser.html has required elements

## Access Points

- **Integrated**: http://localhost:port/browser.html (with site header/footer)
- **Standalone**: http://localhost:port/browser/index.html (full-page browser)

## Features

- ✅ Faceted search with sidebar filters
- ✅ Result cards with thumbnails
- ✅ Click cards to open detailed modals
- ✅ Download YAML files
- ✅ Model Builder for creating simulation experiments
- ✅ Cross-references between models and studies
- ✅ LaTeX equation rendering
- ✅ Markdown description rendering

## Configuration

### _quarto.yml
```yaml
post-render:
  - python3 scripts/generate_browser.py
```

### Data Sources
- `database/models/` - Model definitions
- `database/studies/` - Study metadata
- `database/atlases/` - Brain atlases
- `database/networks/` - Connectivity networks
- `database/coupling_functions/` - Coupling functions
- `schema/` - Schema definitions

## Troubleshooting

### Browser shows "Loading..."
- Check browser console for JavaScript errors
- Verify `data.js` and `schema.js` exist in `_site/browser/`
- Ensure `appTitle` element exists in browser.html

### Cards not clickable
- Verify `cursor: pointer` is in `styles.css` for `.result-card`
- Check that `browser.js` (not simplified version) is loaded

### Missing thumbnails/reports
- Install tvbo: `pip install tvbo`
- Install matplotlib: `pip install matplotlib`
- Thumbnails and reports are optional features

## Development

To update the browser:
1. Edit files in `docs/browser/`
2. Run `quarto render browser.qmd`
3. The post-render script automatically copies updates to `_site/browser/`

No manual file copying needed!
