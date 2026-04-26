#!/usr/bin/env python3
"""
Pre-create tourism POI GeoPackage from filtered CSV.

This script converts the already-clipped and pre-projected POI CSV
to GeoPackage format, avoiding in-notebook conversion delays.

Usage:
    python prepare_poi_gpkg.py
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

# Paths
CSV_PATH = Path('Airbnb/tourism_data/filtered/london_strict_tourism_poi.csv')
GPKG_PATH = Path('Airbnb/tourism_data/filtered/london_strict_tourism_poi.gpkg')

# Fallback for execution from repo root
if not CSV_PATH.exists():
    CSV_PATH = Path('tourism_data/filtered/london_strict_tourism_poi.csv')
    GPKG_PATH = Path('tourism_data/filtered/london_strict_tourism_poi.gpkg')

print(f"Reading POI CSV from: {CSV_PATH}")
print(f"Target GeoPackage: {GPKG_PATH}")

# Read CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {len(df):,} POI records")

# Create geometry from British National Grid coordinates (EPSG:27700)
geometry = [Point(xy) for xy in zip(df['easting'], df['northing'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:27700')

# Select relevant columns for GeoPackage
keep_cols = [
    'poi_id', 'poi_name', 'category', 'category_alt',
    'address', 'locality', 'postcode', 'source',
    'latitude', 'longitude', 'easting', 'northing',
    'lsoa21cd', 'geometry'
]
gdf = gdf[[c for c in keep_cols if c in gdf.columns]]

# Write GeoPackage
gdf.to_file(GPKG_PATH, driver='GPKG', index=False)
print(f"✓ GeoPackage created: {GPKG_PATH}")
print(f"  CRS: {gdf.crs}")
print(f"  Records: {len(gdf):,}")
print(f"  Columns: {', '.join(gdf.columns.tolist())}")
