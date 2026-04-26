#!/usr/bin/env python3
"""
CASA0006 Airbnb ML Pipeline - WITH TOURISM FEATURES
====================================================
This script runs the complete ML pipeline with tourism clustering features.
Results are compared with baseline version to assess tourism impact.

Run: python run_ml_with_tourism.py
"""

import os
import sys
import time
import warnings
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from pyproj import Transformer
from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
LISTINGS_PATH = 'listings.csv'
POI_PATH = 'tourism_data/filtered/london_strict_tourism_poi.csv'
RESULTS_DIR = Path('results_with_tourism')
RESULTS_DIR.mkdir(exist_ok=True)

# DBSCAN Parameters
DBSCAN_EPS = 500  # meters
DBSCAN_MIN_SAMPLES = 15

# ML Parameters
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5

# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================
print("=" * 80)
print("STEP 1: Loading and cleaning data...")
print("=" * 80)

listings = pd.read_csv(LISTINGS_PATH)
print(f"Loaded listings: {len(listings):,} rows")

# Clean: remove permanently delisted
listings = listings[listings['availability_365'] > 0].copy()
print(f"After cleaning: {len(listings):,} rows")

# Load POI
poi = pd.read_csv(POI_PATH)
print(f"Loaded POIs: {len(poi):,} rows")

# ============================================================================
# STEP 2: PREPARE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Preparing features...")
print("=" * 80)

def clean_money(series):
    return (
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        .astype(float)
    )

def parse_bathrooms(bathrooms, bathrooms_text):
    if pd.notna(bathrooms):
        return float(bathrooms)
    if pd.isna(bathrooms_text):
        return np.nan
    import re
    match = re.search(r'(\d+(?:\.\d+)?)', str(bathrooms_text))
    return float(match.group(1)) if match else np.nan

def parse_superhost(series):
    return series.map({'t': 1, 'f': 0, True: 1, False: 0})

listings = listings.dropna(subset=['id', 'latitude', 'longitude', 'estimated_occupancy_l365d']).copy()
listings['price_clean'] = clean_money(listings['price'])
listings['bathrooms_num'] = [parse_bathrooms(b, bt) for b, bt in zip(listings['bathrooms'], listings['bathrooms_text'])]
listings['host_is_superhost_num'] = parse_superhost(listings['host_is_superhost'])

occupancy_threshold = listings['estimated_occupancy_l365d'].quantile(0.75)
listings['high_activity'] = (listings['estimated_occupancy_l365d'] >= occupancy_threshold).astype(int)

print(f"Target: high_activity (threshold={occupancy_threshold:.1f} days/year)")
print(f"  High-activity listings: {listings['high_activity'].sum():,} ({listings['high_activity'].mean()*100:.1f}%)")

# ============================================================================
# STEP 3: SPATIAL FEATURES WITH TOURISM CLUSTERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Computing tourism clustering features...")
print("=" * 80)

wgs84_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

start_spatial = time.time()

# Use lat/lon directly for faster distance calculation
print("  Preparing spatial data...")
sys.stdout.flush()

# Get listing coordinates
listing_coords = listings[['latitude', 'longitude']].values
print(f"  Listings: {len(listing_coords):,} points")
sys.stdout.flush()

# Get POI coordinates
poi_clean = poi.dropna(subset=['easting', 'northing', 'latitude', 'longitude']).copy()
poi_clean = poi_clean.loc[np.isfinite(poi_clean[['easting', 'northing', 'latitude', 'longitude']]).all(axis=1)].copy()
print(f"  POIs: {len(poi_clean):,} points")
sys.stdout.flush()

# Use lat/lon coordinates for cKDTree (fast Euclidean distance)
poi_coords_latlon = poi_clean[['latitude', 'longitude']].values

# DBSCAN Clustering (use BNG coordinates which are already in meters)
print(f"  Clustering POIs with DBSCAN (eps={DBSCAN_EPS}m, min_samples={DBSCAN_MIN_SAMPLES})...")
sys.stdout.flush()
start_dbscan = time.time()

poi_coords_bng = poi_clean[['easting', 'northing']].values
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
poi_cluster_ids = dbscan.fit_predict(poi_coords_bng)
poi_clean['cluster_id'] = poi_cluster_ids

n_clusters = len(set(poi_cluster_ids)) - (1 if -1 in poi_cluster_ids else 0)
n_noise = (poi_cluster_ids == -1).sum()

print(f"    Found {n_clusters} clusters + {n_noise} noise points in {time.time() - start_dbscan:.1f}s")
sys.stdout.flush()

# Compute cluster density
cluster_density = {}
for cid in set(poi_cluster_ids):
    if cid != -1:
        cluster_density[cid] = (poi_cluster_ids == cid).sum()

poi_clean['cluster_density'] = poi_clean['cluster_id'].apply(lambda x: cluster_density.get(x, 0))

# Build spatial index
print("  Building spatial index...")
sys.stdout.flush()
poi_tree = cKDTree(poi_coords_latlon)
print("  ✓ Spatial index built")
sys.stdout.flush()

# Query nearest POIs
print("  Querying nearest POIs...")
sys.stdout.flush()
distances_deg, nearest_idx = poi_tree.query(listing_coords, k=1)

# Convert degree distances to approximate meters
lat_mean = np.mean(listing_coords[:, 0])
lon_factor = np.cos(np.radians(lat_mean))
distances_m = distances_deg * np.sqrt((111000)**2 + (111000 * lon_factor)**2) / np.sqrt(2)

listings['dist_to_nearest_poi'] = distances_m
listings['nearest_poi_category'] = poi_clean['category'].iloc[nearest_idx].values
listings['nearest_cluster_id'] = poi_clean['cluster_id'].iloc[nearest_idx].values
listings['cluster_poi_density'] = poi_clean['cluster_density'].iloc[nearest_idx].values

# Define hotspots (top 25% density)
if n_clusters > 0:
    density_threshold = np.percentile(list(cluster_density.values()), 75)
    hotspot_clusters = set([cid for cid, dens in cluster_density.items() if dens >= density_threshold])
    listings['is_tourism_hotspot'] = listings['nearest_cluster_id'].apply(lambda x: 1 if x in hotspot_clusters else 0)
    print(f"  Tourism hotspots (top 25%): {len(hotspot_clusters)} clusters")
    print(f"    {listings['is_tourism_hotspot'].sum():,} listings near hotspots ({listings['is_tourism_hotspot'].mean()*100:.1f}%)")
else:
    listings['is_tourism_hotspot'] = 0

elapsed_spatial = time.time() - start_spatial
print(f"✓ Tourism features computed in {elapsed_spatial:.1f}s")
print(f"  Median distance to nearest POI: {listings['dist_to_nearest_poi'].median():.0f}m")
sys.stdout.flush()

# ============================================================================
# STEP 3.5: SAVE CLEANED LISTINGS WITH TOURISM FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3.5: Saving cleaned and enriched listings data with tourism features...")
print("=" * 80)

cleaned_listings_path = RESULTS_DIR / 'cleaned_listings_with_tourism_features.csv'
listings.to_csv(cleaned_listings_path, index=False)
print(f"✓ Saved: {cleaned_listings_path}")
print(f"  Columns: {len(listings.columns)} | Rows: {len(listings):,}")
print(f"  New columns: price_clean, bathrooms_num, host_is_superhost_num, dist_to_nearest_poi, nearest_poi_category")
print(f"  Tourism columns: nearest_cluster_id, cluster_poi_density, is_tourism_hotspot")
sys.stdout.flush()

# Save ML-only baseline-compatible features
ml_baseline_path = RESULTS_DIR / 'ml_features_only_baseline_compatible.csv'
numeric_features_baseline = ['price_clean', 'accommodates', 'bedrooms', 'bathrooms_num', 
                             'minimum_nights', 'host_is_superhost_num', 'dist_to_nearest_poi']
categorical_features_baseline = ['room_type', 'neighbourhood_cleansed']
target_baseline = 'high_activity'

listings_ml_baseline = listings[numeric_features_baseline + categorical_features_baseline + [target_baseline]].copy()
listings_ml_baseline.to_csv(ml_baseline_path, index=False)
print(f"\n✓ Saved (ML baseline features): {ml_baseline_path}")
print(f"  Columns: {len(listings_ml_baseline.columns)} ({len(numeric_features_baseline)} numeric + {len(categorical_features_baseline)} categorical + 1 target)")
print(f"  Rows: {len(listings_ml_baseline):,}")

# Save ML-only with tourism features
ml_tourism_path = RESULTS_DIR / 'ml_features_with_tourism.csv'
numeric_features_tourism = ['price_clean', 'accommodates', 'bedrooms', 'bathrooms_num', 
                            'minimum_nights', 'host_is_superhost_num', 
                            'dist_to_nearest_poi', 'cluster_poi_density', 'is_tourism_hotspot']
categorical_features_tourism = ['room_type', 'neighbourhood_cleansed', 'nearest_poi_category']

listings_ml_tourism = listings[numeric_features_tourism + categorical_features_tourism + [target_baseline]].copy()
listings_ml_tourism.to_csv(ml_tourism_path, index=False)
print(f"\n✓ Saved (ML with tourism): {ml_tourism_path}")
print(f"  Columns: {len(listings_ml_tourism.columns)} ({len(numeric_features_tourism)} numeric + {len(categorical_features_tourism)} categorical + 1 target)")
print(f"  Rows: {len(listings_ml_tourism):,}")
sys.stdout.flush()

# ============================================================================
# STEP 4: PREPARE ML FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Preparing ML features...")
print("=" * 80)

numeric_features = [
    'price_clean', 'accommodates', 'bedrooms', 'bathrooms_num', 
    'minimum_nights', 'host_is_superhost_num', 
    'dist_to_nearest_poi', 'cluster_poi_density', 'is_tourism_hotspot'  # Tourism features
]
categorical_features = ['room_type', 'neighbourhood_cleansed', 'nearest_poi_category']
target = 'high_activity'

df_ml = listings[numeric_features + categorical_features + [target]].dropna(subset=[target]).copy()

print(f"Final dataset: {len(df_ml):,} rows")
print(f"  Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")
print(f"  Tourism features: dist_to_nearest_poi, cluster_poi_density, is_tourism_hotspot")
print(f"  High-activity: {df_ml[target].sum():,} ({df_ml[target].mean()*100:.1f}%)")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Splitting data...")
print("=" * 80)

X = df_ml[numeric_features + categorical_features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train high-activity: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

# ============================================================================
# STEP 6: PREPROCESSING PIPELINE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Building preprocessing pipeline...")
print("=" * 80)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ],
    remainder='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed features: {X_train_processed.shape[1]}")

# ============================================================================
# STEP 7: MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Training models with GridSearchCV...")
print("=" * 80)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

# Model 1: Logistic Regression
print("\n[1/3] Logistic Regression...")
start_model = time.time()
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['lbfgs']}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                       lr_params, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
lr_grid.fit(X_train_processed, y_train)
results['logistic_regression'] = {
    'model': lr_grid.best_estimator_,
    'best_params': lr_grid.best_params_,
    'cv_score': lr_grid.best_score_,
    'training_time': time.time() - start_model
}
print(f"  Best params: {lr_grid.best_params_} | CV F1: {lr_grid.best_score_:.4f} | Time: {results['logistic_regression']['training_time']:.1f}s")

# Model 2: Random Forest
print("[2/3] Random Forest...")
start_model = time.time()
rf_params = {'n_estimators': [50, 100], 'max_depth': [10, 15, 20], 'min_samples_split': [5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                       rf_params, cv=cv, scoring='f1', n_jobs=1, verbose=0)
rf_grid.fit(X_train_processed, y_train)
results['random_forest'] = {
    'model': rf_grid.best_estimator_,
    'best_params': rf_grid.best_params_,
    'cv_score': rf_grid.best_score_,
    'training_time': time.time() - start_model
}
print(f"  Best params: {rf_grid.best_params_} | CV F1: {rf_grid.best_score_:.4f} | Time: {results['random_forest']['training_time']:.1f}s")

# Model 3: XGBoost
print("[3/3] XGBoost...")
start_model = time.time()
xgb_params = {'max_depth': [5, 7, 10], 'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]}
xgb_grid = GridSearchCV(XGBClassifier(random_state=RANDOM_STATE, verbosity=0),
                        xgb_params, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train_processed, y_train)
results['xgboost'] = {
    'model': xgb_grid.best_estimator_,
    'best_params': xgb_grid.best_params_,
    'cv_score': xgb_grid.best_score_,
    'training_time': time.time() - start_model
}
print(f"  Best params: {xgb_grid.best_params_} | CV F1: {xgb_grid.best_score_:.4f} | Time: {results['xgboost']['training_time']:.1f}s")

# ============================================================================
# STEP 8: EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Evaluating models on test set...")
print("=" * 80)

evaluation = {}
for name, result in results.items():
    model = result['model']
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    evaluation[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name.upper()}")
    for metric, value in evaluation[name].items():
        print(f"  {metric}: {value:.4f}")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Saving results...")
print("=" * 80)

eval_df = pd.DataFrame(evaluation).T
eval_df.to_csv(RESULTS_DIR / 'with_tourism_evaluation.csv')
print(f"✓ Saved: {RESULTS_DIR / 'with_tourism_evaluation.csv'}")

with open(RESULTS_DIR / 'with_tourism_models.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"✓ Saved: {RESULTS_DIR / 'with_tourism_models.pkl'}")

metadata = {
    'version': 'with_tourism',
    'listings_count': len(listings),
    'pois_count': len(poi_clean),
    'dbscan_eps': DBSCAN_EPS,
    'dbscan_min_samples': DBSCAN_MIN_SAMPLES,
    'n_tourism_clusters': n_clusters,
    'features_numeric': numeric_features,
    'features_categorical': categorical_features,
    'target': target,
    'high_activity_threshold': occupancy_threshold,
    'test_size': TEST_SIZE,
    'cv_folds': CV_FOLDS,
    'random_state': RANDOM_STATE,
    'timestamp': pd.Timestamp.now().isoformat()
}
with open(RESULTS_DIR / 'with_tourism_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved: {RESULTS_DIR / 'with_tourism_metadata.json'}")

print("\n" + "=" * 80)
print("✅ WITH-TOURISM ML PIPELINE COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
print("\nNext step: Run python compare_results.py to compare baseline vs tourism-enhanced")
