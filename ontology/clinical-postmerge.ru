PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos:    <http://www.w3.org/2004/02/skos/core#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX owl:     <http://www.w3.org/2002/07/owl#>
PREFIX tvbc:    <https://w3id.org/tvbo/clinical/>

# Drop the external NIF atom.ttl import (owlready2 cannot fetch it; owl.py strips it anyway).
DELETE { ?o owl:imports <https://raw.githubusercontent.com/SciCrunch/NIF-Ontology/atlas/ttl/atom.ttl> }
WHERE  { ?o owl:imports <https://raw.githubusercontent.com/SciCrunch/NIF-Ontology/atlas/ttl/atom.ttl> } ;

# Make clinical entities findable by the extension's label search (owlready2 onto.search(label=)
# matches rdfs:label, but the clinical addon labels domains/models/ICD chapters with skos:prefLabel).
INSERT { ?s rdfs:label ?l }
WHERE  { ?s skos:prefLabel ?l . FILTER NOT EXISTS { ?s rdfs:label ?any } } ;

# Label clinical studies by their citekey (study-Adams2022 -> "Adams2022") so they show cleanly.
INSERT { ?s rdfs:label ?key }
WHERE  { ?s a tvbc:ClinicalApplicationStudy .
         BIND(STRAFTER(STR(?s), "clinical/study-") AS ?key)
         FILTER(?key != "") FILTER NOT EXISTS { ?s rdfs:label ?any } }
