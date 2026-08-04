# Data — provenance, licenses, filters

This directory holds dataset loaders, custom clinical vignettes, and subspecialty
filters. **No raw clinical data is committed.** Loaders pull from public sources
at run-time; vignettes are textbook-style and demographic-anonymized.

## Datasets used in v1.0

| Dataset          | Source (HF)                   | Filter rationale                               | License   |
| ---------------- | ----------------------------- | ---------------------------------------------- | --------- |
| MedMCQA          | `medmcqa`                     | topic ∈ {cardiology, rheumatology, immunology} | MIT       |
| MedQA-USMLE      | `bigbio/med_qa`               | regex on cardio + autoimmune keywords          | MIT       |
| PubMedQA         | `pubmed_qa`                   | MeSH terms cardio + autoimmune                 | MIT       |
| Custom Vignettes | `data/vignettes/` (this repo) | hand-written, two-reviewer rule                | CC-BY-4.0 |

Filter scripts live in `data/filters/` and the rationale for each filter is documented in `notebooks/02_subspecialty_filtering.ipynb` (added in Phase 2).
