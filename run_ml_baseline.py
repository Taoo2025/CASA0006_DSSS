#!/usr/bin/env python3
"""
CASA0006 Airbnb ML Pipeline - BASELINE VERSION (No Tourism Features)
=====================================================================
This script runs the complete ML pipeline with baseline features only.
Results are saved for comparison with tourism-feature version.

Run: python run_ml_baseline.py
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
from pyproj import Transformer

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
RESULTS_DIR = Path('results_baseline')
RESULTS_DIR.mkdir(exist_ok=True)

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

# Clean: remove permanently delisted (availability_365 = 0)
listings = listings[listings['availability_365'] > 0].copy()
print(f"After cleaning: {len(listings):,} rows (removed {len(listings) - (listings['availability_365'] > 0).sum():,} delisted)")

# Load POI
poi = pd.read_csv(POI_PATH)
print(f"Loaded POIs: {len(poi):,} rows")

# ============================================================================
# STEP 2: PREPARE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Preparing features...")
print("=" * 80)

# Define helper functions
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

# Clean listings
listings = listings.dropna(subset=['id', 'latitude', 'longitude', 'estimated_occupancy_l365d']).copy()
listings['price_clean'] = clean_money(listings['price'])
listings['bathrooms_num'] = [parse_bathrooms(b, bt) for b, bt in zip(listings['bathrooms'], listings['bathrooms_text'])]
listings['host_is_superhost_num'] = parse_superhost(listings['host_is_superhost'])

# Define target: high_activity (top 25% occupancy)
occupancy_threshold = listings['estimated_occupancy_l365d'].quantile(0.75)
listings['high_activity'] = (listings['estimated_occupancy_l365d'] >= occupancy_threshold).astype(int)

print(f"Target variable: high_activity (threshold={occupancy_threshold:.1f} days/year)")
print(f"  High-activity listings: {listings['high_activity'].sum():,} ({listings['high_activity'].mean()*100:.1f}%)")

# ============================================================================
# STEP 3: SPATIAL FEATURES (BASELINE: Nearest POI category) - FAST VERSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Computing baseline spatial features (fast vectorized)...")
print("=" * 80)
sys.stdout.flush()

start_spatial = time.time()

# Use lat/lon directly for faster distance calculation
# Convert POI BNG coordinates back to lat/lon using direct formula
# (This is faster than transforming all listings)
print("  Preparing spatial data...")
sys.stdout.flush()

# Get listing coordinates
listing_coords = listings[['latitude', 'longitude']].values
print(f"  Listings: {len(listing_coords):,} points")
sys.stdout.flush()

# Get POI coordinates - convert BNG back to lat/lon approximation
# Using inverse projection approximation (good enough for nearest neighbor)
poi_clean = poi.dropna(subset=['easting', 'northing', 'latitude', 'longitude']).copy()
poi_clean = poi_clean.loc[np.isfinite(poi_clean[['easting', 'northing', 'latitude', 'longitude']]).all(axis=1)].copy()
print(f"  POIs: {len(poi_clean):,} points")
sys.stdout.flush()

# Use lat/lon coordinates for cKDTree (fast Euclidean distance)
# At London's latitude (~51.5°), 1° lat ≈ 111 km, 1° lon ≈ 72 km
# So Euclidean distance on (lat, lon) gives approximate distances
# For nearest neighbor finding, this ordering is correct
poi_coords_latlon = poi_clean[['latitude', 'longitude']].values

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
# At latitude 51.5°: 1° lat = 111 km, 1° lon = 111*cos(51.5°) = 111*0.62 = 69 km
# Approximate conversion: distance_m = distance_deg * 111000 (simplified)
lat_mean = np.mean(listing_coords[:, 0])
lon_factor = np.cos(np.radians(lat_mean))
distances_m = distances_deg * np.sqrt((111000)**2 + (111000 * lon_factor)**2) / np.sqrt(2)

listings['dist_to_nearest_poi'] = distances_m
listings['nearest_poi_category'] = poi_clean['category'].iloc[nearest_idx].values

elapsed_spatial = time.time() - start_spatial
print(f"✓ Spatial features computed in {elapsed_spatial:.1f}s")
print(f"  Median distance to nearest POI: {listings['dist_to_nearest_poi'].median():.0f}m")
sys.stdout.flush()

# ============================================================================
# STEP 3.5: SAVE CLEANED LISTINGS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3.5: Saving cleaned and enriched listings data...")
print("=" * 80)

cleaned_listings_path = RESULTS_DIR / 'cleaned_listings_with_features.csv'
listings.to_csv(cleaned_listings_path, index=False)
print(f"✓ Saved: {cleaned_listings_path}")
print(f"  Columns: {len(listings.columns)} | Rows: {len(listings):,}")
print(f"  New columns added: price_clean, bathrooms_num, host_is_superhost_num, dist_to_nearest_poi, nearest_poi_category, high_activity")
sys.stdout.flush()

# Save ML-only features subset
ml_features_path = RESULTS_DIR / 'ml_features_only.csv'
numeric_features_save = ['price_clean', 'accommodates', 'bedrooms', 'bathrooms_num', 
                         'minimum_nights', 'host_is_superhost_num', 'dist_to_nearest_poi']
categorical_features_save = ['room_type', 'neighbourhood_cleansed']
target_save = 'high_activity'

listings_ml = listings[numeric_features_save + categorical_features_save + [target_save]].copy()
listings_ml.to_csv(ml_features_path, index=False)
print(f"\n✓ Saved (ML features only): {ml_features_path}")
print(f"  Columns: {len(listings_ml.columns)} ({len(numeric_features_save)} numeric + {len(categorical_features_save)} categorical + 1 target)")
print(f"  Rows: {len(listings_ml):,}")
sys.stdout.flush()

# ============================================================================
# STEP 4: PREPARE ML FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Preparing ML features...")
print("=" * 80)

# Select features
numeric_features = ['price_clean', 'accommodates', 'bedrooms', 'bathrooms_num', 
                    'minimum_nights', 'host_is_superhost_num', 'dist_to_nearest_poi']
categorical_features = ['room_type', 'neighbourhood_cleansed', 'nearest_poi_category']
target = 'high_activity'

# Drop rows with missing target
df_ml = listings[numeric_features + categorical_features + [target]].dropna(subset=[target]).copy()

print(f"Final dataset: {len(df_ml):,} rows, {len(numeric_features)} numeric + {len(categorical_features)} categorical features")
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
print(f"Train high-activity rate: {y_train.mean()*100:.1f}%")
print(f"Test high-activity rate: {y_test.mean()*100:.1f}%")

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

# Save evaluation
eval_df = pd.DataFrame(evaluation).T
eval_df.to_csv(RESULTS_DIR / 'baseline_evaluation.csv')
print(f"✓ Saved: {RESULTS_DIR / 'baseline_evaluation.csv'}")

# Save models
with open(RESULTS_DIR / 'baseline_models.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"✓ Saved: {RESULTS_DIR / 'baseline_models.pkl'}")

# Save metadata
metadata = {
    'version': 'baseline',
    'listings_count': len(listings),
    'pois_count': len(poi_clean),
    'features_numeric': numeric_features,
    'features_categorical': categorical_features,
    'target': target,
    'high_activity_threshold': occupancy_threshold,
    'test_size': TEST_SIZE,
    'cv_folds': CV_FOLDS,
    'random_state': RANDOM_STATE,
    'timestamp': pd.Timestamp.now().isoformat()
}
with open(RESULTS_DIR / 'baseline_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved: {RESULTS_DIR / 'baseline_metadata.json'}")

print("\n" + "=" * 80)
print("✅ BASELINE ML PIPELINE COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
print("\nNext step: Run run_ml_with_tourism.py for comparison")
