# BIDS Computational Modelling (BEP034)

This document describes how computational modelling data are intended to be
represented within the Brain Imaging Data Structure (BIDS) as proposed in
BIDS Extension Proposal 034 (BEP034).

## Relationship matrices

Relationship matrices (for example, subject–subject, run–run, or model–model
relationships) are defined and standardised in BIDS Extension Proposal 017
(BEP017). Implementations that need to store or reference relationship
matrices as part of a BEP034-compliant dataset **must** follow the BEP017
specification for:

- File naming and directory placement of relationship matrix files.
- The tabular structure (columns, required and optional fields).
- Admissible value ranges and encoding conventions.

BEP034 therefore focuses on modelling-specific entities (such as model
descriptions, parameter estimates, and prediction outputs), while delegating
the representation of generic relationship matrices to BEP017. When
documenting or implementing computational modelling datasets:

- Use BEP034 for all modelling-specific structures and metadata.
- Use BEP017 wherever relationship matrices are involved, and reference those
  files from BEP034 entities as appropriate.

For the complete and authoritative description of relationship matrices, refer
to the BEP017 specification in the BIDS documentation.
