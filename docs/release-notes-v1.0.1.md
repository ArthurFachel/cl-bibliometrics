Archival release for Zenodo (adds a DOI) with small corrections. No analysis or result changed since v1.0.0; every file in results/ is identical.

Changes
- CITATION.cff added (author, ORCID, abstract), so GitHub and Zenodo show how to cite the work.
- Report byline: Data Science, PUCRS (was Data Analytics).
- validation/nlp_round2_prereg.md and a comment in src/clbib/benchmarks.py: round 2 was scored on 2026-10-20, not 2026-10-21.
- README: drivers figure first; links point to the latest release.

Findings and method: see the v1.0.0 notes and the README.

Assets
- report.pdf: 3-page summary.
- cl-bibliometrics-processed-2026-10-19.zip: same file as in v1.0.0 (sha256 05d7d5ba006f406f50fe801d74fc11ef183e62b9c68bdf09ddd8731884feda1c). Unzip at the repo root and run any make target except fetch and clean, with no API key.
