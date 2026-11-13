PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# Every TVB-O term is defined by the merged ontology, whichever module declares it.
INSERT { ?term rdfs:isDefinedBy <https://w3id.org/tvbo/tvbo.owl> }
WHERE  {
  ?term a ?type .
  FILTER (isIRI(?term) && STRSTARTS(STR(?term), "https://w3id.org/tvbo/"))
  FILTER NOT EXISTS { ?term a owl:Ontology }
}
