# CASA0006 · Data Science for Spatial Systems — Final Project

MSc coursework at [CASA, UCL](https://www.ucl.ac.uk/bartlett/casa).

## Predicting Vulnerable Road User (VRU) Collision Severity in London

Machine-learning analysis of **DfT road casualty statistics** (5 years, London), focusing on pedestrians and cyclists.

- Built borough-month feature sets from collision, casualty and vehicle records
- Engineered spatial and contextual features (incl. tourism POI density from OSM)
- Trained and compared ML classifiers for collision severity; interpreted feature importance
- Mapped risk patterns at borough and LSOA level

## Key files

| File | Description |
|---|---|
| `vru_collision_severity_analysis.ipynb` | Main analysis notebook |
| `run_ml_with_tourism.py` | ML pipeline with tourism features |
| `prepare_poi_gpkg.py` | OSM tourism POI preparation |
| `London_*.csv`, `London_Boroughs.gpkg` | Processed input data |

## Tools
Python · pandas · GeoPandas · scikit-learn · matplotlib
