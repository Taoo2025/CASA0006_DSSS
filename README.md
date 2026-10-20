# London Airbnb Activity Prediction
### Do tourism-related spatial factors improve prediction?

A machine learning and geospatial analysis project for **CASA0006 · Data Science for Spatial Systems, UCL**.

Compare listing characteristics with additional tourism-cluster features to classify high-activity Airbnb listings in London. The target is the top quartile of Inside Airbnb's estimated occupancy measure (approximately 92 days/year in the submitted analysis), **a proxy for relative activity rather than observed bookings**.

## Explore the project

- **[Final analysis notebook](airbnb_activity_analysis.ipynb)** — research narrative, model comparison, ROC/PR curves, SHAP, permutation importance and borough analysis.
- [Recorded model results](results/model_comparison.csv) — values preserved from the submitted notebook outputs.
- [Tourism feature preparation and training](run_ml_with_tourism.py) — spatial feature construction and a separate training workflow.
- [Original baseline training script](run_ml_baseline.py) and [script-results comparison](compare_results.py).

## Approach

1. Clean listing attributes and define the high-activity classification target.
2. Project tourism POIs into British National Grid (EPSG:27700); identify clusters using DBSCAN (500 m radius, minimum 15 POIs).
3. Compare Logistic Regression, Random Forest and XGBoost using a stratified 75/25 train/test split, seed 42, and five-fold stratified tuning scored by F1.
4. Evaluate precision, recall, F1 and ROC-AUC; interpret predictions using SHAP and permutation importance.

**What the final notebook actually compares:** the baseline contains price, capacity, bedrooms, bathrooms, minimum nights, Superhost status, room type, borough **and distance to the nearest tourism POI**. The enhanced feature set adds cluster POI density and a tourism-hotspot indicator. Thus this experiment measures the incremental value of clustering features; it is not a comparison against a completely tourism-free baseline. The standalone scripts use a different feature configuration and should be treated as separate experiments.

## Recorded results

| Model | Baseline F1 | Enhanced F1 | Baseline ROC-AUC | Enhanced ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.512804 | 0.521524 | 0.735051 | 0.736503 |
| Random Forest | 0.570522 | 0.571734 | 0.787557 | **0.792765** |
| XGBoost | 0.573366 | **0.573508** | 0.786366 | 0.788571 |

XGBoost has the highest held-out F1; Random Forest has the highest ROC-AUC. Cluster features give small gains across these metrics. XGBoost's F1 increases by **0.000142**, so the result should not be presented as a large performance improvement. Listing attributes, particularly Superhost status and price, remain prominent in the interpretation.

These values are transcribed from existing notebook outputs, not a newly executed training run.

## Run the final notebook

Use Python 3.10+ in a fresh environment:

```bash
python -m pip install -r requirements.txt
jupyter lab airbnb_activity_analysis.ipynb
```

Run cells in order. The notebook downloads the two prepared feature tables from this repository's main branch. Keep `London_Boroughs.gpkg` in the working directory for the borough map. Model tuning can take several minutes; runtime depends on hardware. The dependency list is unpinned because the original environment was not locked.

### Rebuild features from source data

The raw listings and full UK POI dataset are not added in this update.

- Obtain London listings with `estimated_occupancy_l365d` from [Inside Airbnb](https://insideairbnb.com/get-the-data/). The original snapshot date was not recorded in the supplied final report.
- Tourism POIs originate from the [GeoDS UK POI dataset](https://data.geods.ac.uk/dataset/point-of-interest-data-for-the-united-kingdom), filtered to Greater London and tourism categories. Observe the providers' data terms.
- Put `listings.csv` in the working directory and copy the existing `london_strict_tourism_poi.csv` into `tourism_data/filtered/`.
- Run `python run_ml_baseline.py`, then `python run_ml_with_tourism.py`, then `python compare_results.py` for the standalone-script experiment. These outputs are not expected to match the final notebook's table because the feature definitions differ.
- `prepare_poi_gpkg.py` converts an already filtered CSV to a GeoPackage; it does not implement the full raw UK POI filtering workflow.

## Interpretation and limitations

- Estimated occupancy is a model-derived proxy. Predictions do not establish causal effects on bookings or housing markets.
- Random listing splits do not test transfer to unseen geographic areas or future periods.
- The submitted notebook fits preprocessing before cross-validation; tuning scores may be optimistic. A future evaluation should put preprocessing inside each cross-validation fold.
- The small measured gains do not establish statistical significance.
- The added notebook retains original successful outputs. The borough boundary loading step was repaired, but the full analysis has not been rerun in this update.

## Earlier exploration

The existing `vru_collision_severity_analysis.ipynb` and `London_collision.csv`, `London_casualty.csv`, `London_vehicle.csv` belong to an earlier road-safety exploration. They are retained for provenance and are not the final Airbnb project. `London_Boroughs.gpkg` is also used by the Airbnb mapping step.

**Tools:** Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · GeoPandas · pyproj · Matplotlib · seaborn · SHAP
