PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX tvbo: <https://w3id.org/tvbo/>

# tvbo:model_type and tvbo:unit are LinkML enum-ranged slots that LinkML emits as
# owl:ObjectProperty, but the A-box uses string literals as values. OWLAPI/Widoco
# crashes trying to cast those literals to IRIs. Re-type them as AnnotationProperty
# and drop the (now OWL-DL-incompatible) restrictions referencing them.

# 1. Re-type as AnnotationProperty
DELETE { tvbo:model_type a owl:ObjectProperty }
INSERT { tvbo:model_type a owl:AnnotationProperty }
WHERE  { tvbo:model_type a owl:ObjectProperty } ;

DELETE { tvbo:unit a owl:ObjectProperty }
INSERT { tvbo:unit a owl:AnnotationProperty }
WHERE  { tvbo:unit a owl:ObjectProperty } ;

# 2. Detach restriction blanks from their parent classes and clear their contents
DELETE { ?cls ?ax ?r . ?r ?p ?o }
WHERE  {
  ?r owl:onProperty tvbo:model_type ;
     ?p ?o .
  OPTIONAL { ?cls ?ax ?r }
} ;

DELETE { ?cls ?ax ?r . ?r ?p ?o }
WHERE  {
  ?r owl:onProperty tvbo:unit ;
     ?p ?o .
  OPTIONAL { ?cls ?ax ?r }
} ;
