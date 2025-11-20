window.searchSchema = {
  "title": "TVB-O Browser",
  "description": "Search and explore TVB-O models, atlases, studies, and networks.",
  "searchPlaceholder": "Search names, titles, DOI…",
  "searchableFields": [
    "name",
    "title",
    "description",
    "label",
    "doi",
    "type"
  ],
  "facets": [
    {
      "field": "type",
      "label": "Type",
      "type": "string",
      "sortBy": "count"
    },
    {
      "field": "year",
      "label": "Year",
      "type": "string",
      "sortBy": "count"
    }
  ],
  "displayFields": [
    {
      "field": "name",
      "label": "Name",
      "type": "string"
    },
    {
      "field": "title",
      "label": "Title",
      "type": "string"
    },
    {
      "field": "type",
      "label": "Type",
      "type": "string"
    },
    {
      "field": "description",
      "label": "Description",
      "type": "string"
    },
    {
      "field": "year",
      "label": "Year",
      "type": "string"
    },
    {
      "field": "doi",
      "label": "Doi",
      "type": "string"
    }
  ]
};
window.dispatchEvent(new Event('searchDataReady'));
