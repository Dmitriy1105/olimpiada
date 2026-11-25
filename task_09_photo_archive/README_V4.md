# Task 9 — Фотоархив: V4

V4 starts from the tested V2 submission and adds crop-level distance-signature matching.

For each crop, the solver aggregates known distances to the current V2 anchor groups. Two crops from different groups provide positive evidence when:
- they have at least two common anchor groups;
- the median difference between their distance signatures is below 0.30 m;
- the maximum difference is below 0.90 m;
- their masked visual similarity is at least 0.55.

Crop-pair evidence is aggregated at group level. A maximum bipartite matching estimates how much of the smaller group is supported. Tiered thresholds are stricter for larger groups. A constrained DSU rejects any merge that would violate a saved distance edge.

## Run

    pip install -r requirements.txt
    python solve_v4.py --data /path/to/fragment_album --base . --output submission_v4.csv

The base directory must contain submission_v2.csv; solve.py must be in the same directory.

## QC

The V4 run produced 11,303 rows for 58 scenes and accepted 13 group merges. All 21,986 saved distance constraints were respected. No crop was missing or duplicated.

The dataset does not include ground-truth object groups. Therefore local QC cannot compute official pairwise F1. The official score must be checked by submitting submission_v4.csv using the Make submission mode.
