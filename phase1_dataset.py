# ============================================================
#  PHASE 1 — WILDFIRE DATASET GENERATION
#  Region : Mysuru + Coorg + Chamarajanagara, Karnataka
#  Output : wildfire_mysuru_2022.csv
#
#  HOW TO RUN:
#  python phase1_dataset.py
# ============================================================

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import os, time, io

from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.neighbors import BallTree
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials

print("Imports OK ✅")


# ── 1. AUTHENTICATE & INIT ───────────────────────────────────
# HOW TO FIX "Project not found" error:
# ─────────────────────────────────────
# Option A (Recommended — use your personal project):
#   1. Go to https://console.cloud.google.com/
#   2. Create a new project (note its Project ID)
#   3. Go to https://code.earthengine.google.com/ → click your avatar → Register new Cloud project
#   4. Replace PROJECT_ID below with your actual GCP project ID
#
# Option B (Quick test — no project needed):
#   Set USE_HIGHVOLUME = True below. This uses the high-volume endpoint
#   which doesn't require a project but has lower quotas.
# ─────────────────────────────────────

PROJECT_ID     = 'wildfire-prediction-491719'   # ← CHANGE THIS to your actual GCP project ID
USE_HIGHVOLUME = False               # ← Set True to skip project requirement (testing only)

def init_ee():
    if USE_HIGHVOLUME:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("Earth Engine ready (high-volume endpoint) ✅")
        return

    try:
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine ready ✅")
    except ee.ee_exception.EEException as e:
        if 'not found or deleted' in str(e) or 'USER_PROJECT_DENIED' in str(e):
            print(
                f"\n❌  GCP project '{PROJECT_ID}' not found.\n"
                "   Fix steps:\n"
                "   1. Visit https://console.cloud.google.com/ and create a project\n"
                "   2. Register it at https://code.earthengine.google.com/\n"
                "      (click avatar → Register new Cloud project)\n"
                "   3. Copy the Project ID and paste it into PROJECT_ID above\n"
                "   4. Re-run this script\n\n"
                "   OR set USE_HIGHVOLUME = True above for a quick no-project test.\n"
            )
            raise SystemExit(1)
        else:
            # Token expired / not authenticated yet — re-auth then retry
            print("Authenticating — browser will open...")
            ee.Authenticate()
            try:
                ee.Initialize(project=PROJECT_ID)
                print("Earth Engine ready ✅")
            except ee.ee_exception.EEException as e2:
                if 'not found or deleted' in str(e2) or 'USER_PROJECT_DENIED' in str(e2):
                    print(
                        f"\n❌  GCP project '{PROJECT_ID}' not found even after auth.\n"
                        "   Follow the steps printed above to create & register your project.\n"
                    )
                    raise SystemExit(1)
                raise

init_ee()


# ── 2. STUDY AREA ─────────────────────────────────────────────
LAT_MIN, LAT_MAX = 11.6, 12.6
LON_MIN, LON_MAX = 75.7, 77.0
study_area = ee.Geometry.BBox(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
print(f"Study area: {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E ✅")


# ── 3. HELPER: NEAREST-NEIGHBOUR MERGE ───────────────────────
def nearest_merge(df_base, df_add, value_col, max_dist_deg=0.05):
    base_rad  = np.radians(df_base[['latitude', 'longitude']].values)
    add_rad   = np.radians(df_add[['latitude',  'longitude']].values)
    tree      = BallTree(add_rad, metric='haversine')
    dist, idx = tree.query(base_rad, k=1)
    dist_deg  = np.degrees(dist.flatten())
    df_base   = df_base.copy()
    df_base[value_col] = df_add[value_col].iloc[idx.flatten()].values
    df_base.loc[dist_deg > max_dist_deg, value_col] = np.nan
    return df_base


# ── 4. HELPER: DOWNLOAD CSV FROM GOOGLE DRIVE ────────────────
def download_from_drive(filename):
    """
    Downloads a file from Google Drive using YOUR OAuth credentials.

    ONE-TIME SETUP (do this before running):
      1. https://console.cloud.google.com/apis/credentials?project=wildfire-prediction-491719
         → CREATE CREDENTIALS → OAuth client ID → Desktop app → Download JSON
         → Save as  client_secrets.json  in this folder
      2. https://console.cloud.google.com/apis/library/drive.googleapis.com?project=wildfire-prediction-491719
         → Enable the Drive API
      Then run the script — browser opens once, you approve, done.
    """
    import json
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GoogleAuthRequest

    DRIVE_SCOPES     = ['https://www.googleapis.com/auth/drive.readonly']
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    client_secrets   = os.path.join(script_dir, 'client_secrets.json')
    drive_token_path = os.path.join(script_dir, '.drive_token.json')

    if not os.path.exists(client_secrets):
        raise FileNotFoundError(
            "\n❌  client_secrets.json not found!\n"
            "   Steps:\n"
            "   1. Go to https://console.cloud.google.com/apis/credentials"
            "?project=wildfire-prediction-491719\n"
            "   2. CREATE CREDENTIALS → OAuth client ID → Desktop app\n"
            "   3. Download the JSON and save it as 'client_secrets.json'\n"
            "      in the same folder as this script.\n"
            "   4. Enable Drive API at:\n"
            "      https://console.cloud.google.com/apis/library/"
            "drive.googleapis.com?project=wildfire-prediction-491719\n"
        )

    creds = None

    # Load cached token
    if os.path.exists(drive_token_path):
        with open(drive_token_path) as f:
            creds = Credentials.from_authorized_user_info(json.load(f), DRIVE_SCOPES)

    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
        except Exception:
            creds = None

    # Fresh browser login (only needed once)
    if not creds or not creds.valid:
        flow  = InstalledAppFlow.from_client_secrets_file(client_secrets, DRIVE_SCOPES)
        print("\n  Opening browser for Google Drive access (one-time only)...")
        creds = flow.run_local_server(port=0)
        with open(drive_token_path, 'w') as f:
            f.write(creds.to_json())
        print("  Drive token saved ✅")

    service = build('drive', 'v3', credentials=creds)

    # Search for the file (GEE exports to root of Drive by default)
    results = service.files().list(
        q         = f"name='{filename}' and trashed=false",
        spaces    = 'drive',
        fields    = 'files(id, name)',
        orderBy   = 'createdTime desc'
    ).execute()

    files = results.get('files', [])
    if not files:
        raise FileNotFoundError(
            f"'{filename}' not found in Google Drive.\n"
            f"Check https://drive.google.com — GEE may have saved it there.\n"
            f"Download it manually and place it in this folder, then re-run."
        )

    file_id = files[0]['id']
    print(f"  Found on Drive: {filename} (id={file_id})")

    request  = service.files().get_media(fileId=file_id)
    buf      = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf)


# ── 5. HELPER: EXPORT + DOWNLOAD ─────────────────────────────
def export_and_download(image, description, bands, scale=1000, n_pixels=2000):
    """
    Submit GEE export task → wait → download CSV from Drive → return DataFrame.
    """
    fc = image.sample(
        region    = study_area,
        scale     = scale,
        numPixels = n_pixels,
        seed      = 42,
        geometries= True
    )
    task = ee.batch.Export.table.toDrive(
        collection  = fc,
        description = description,
        fileFormat  = 'CSV',
        selectors   = ['.geo'] + bands
    )
    task.start()
    print(f"  Task '{description}' submitted — waiting", end='', flush=True)

    while task.active():
        time.sleep(15)
        print('.', end='', flush=True)

    status = task.status()
    if status['state'] != 'COMPLETED':
        raise RuntimeError(f"GEE task failed: {status}")
    print(' done ✅')

    # Download from Google Drive
    df = download_from_drive(f'{description}.csv')

    # Parse geometry: GEE writes coords as 'POINT (lon lat)' in .geo column
    if '.geo' in df.columns:
        import json as _json
        def parse_geo(geo_str):
            try:
                g = _json.loads(geo_str)
                coords = g['coordinates']
                return coords[0], coords[1]   # lon, lat
            except Exception:
                return None, None
        df[['longitude','latitude']] = df['.geo'].apply(
            lambda s: pd.Series(parse_geo(s))
        )
        df = df.drop(columns=['.geo'])

    keep = ['latitude', 'longitude'] + [b for b in bands if b in df.columns]
    df   = df[keep].dropna().reset_index(drop=True)
    return df


# ── 6. NDVI ───────────────────────────────────────────────────
print("\n[1/5] NDVI (MODIS MOD13Q1)...")
ndvi_img = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate('2022-01-01', '2022-12-31')
            .filterBounds(study_area)
            .mean().clip(study_area))
df_ndvi = export_and_download(ndvi_img, 'wf_ndvi', ['NDVI'])
df_ndvi['NDVI'] = df_ndvi['NDVI'] / 10000
print(f"  {len(df_ndvi)} rows | {df_ndvi['NDVI'].min():.3f}-{df_ndvi['NDVI'].max():.3f}")


# ── 7. LST ────────────────────────────────────────────────────
print("\n[2/5] LST (MODIS MOD11A2)...")
lst_img = (ee.ImageCollection("MODIS/061/MOD11A2")
           .filterDate('2022-01-01', '2022-12-31')
           .filterBounds(study_area)
           .mean().clip(study_area))
df_lst = export_and_download(lst_img, 'wf_lst', ['LST_Day_1km'])
df_lst = df_lst.rename(columns={'LST_Day_1km': 'LST'})
df_lst['LST'] = (df_lst['LST'] * 0.02) - 273.15
print(f"  {len(df_lst)} rows | {df_lst['LST'].min():.1f}-{df_lst['LST'].max():.1f} C")


# ── 8. MERGE NDVI + LST ───────────────────────────────────────
print("\nMerging NDVI + LST...")
df = nearest_merge(df_ndvi, df_lst, 'LST')
df = df.dropna(subset=['LST']).reset_index(drop=True)
print(f"  {len(df)} rows")


# ── 9. RAINFALL ───────────────────────────────────────────────
print("\n[3/5] Rainfall (CHIRPS)...")
rain_img = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate('2022-01-01', '2022-12-31')
            .mean().clip(study_area))
df_rain = export_and_download(rain_img, 'wf_rain', ['precipitation'])
df_rain = df_rain.rename(columns={'precipitation': 'rainfall'})
df      = nearest_merge(df, df_rain, 'rainfall')
df      = df.dropna(subset=['rainfall']).reset_index(drop=True)
print(f"  {len(df)} rows | mean {df['rainfall'].mean():.2f} mm/day")


# ── 10. POPULATION DENSITY ────────────────────────────────────
print("\n[4/5] Population density (GPWv4)...")
pop_img = (ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
           .filterDate('2020-01-01', '2020-12-31')
           .mean().clip(study_area))
df_pop = export_and_download(pop_img, 'wf_pop', ['population_density'])
df     = nearest_merge(df, df_pop, 'population_density')
df     = df.dropna(subset=['population_density']).reset_index(drop=True)
print(f"  {len(df)} rows")


# ── 11. ROAD PROXIMITY ────────────────────────────────────────
# OSMnx over a 1.3°x1.3° bbox is too large (~13000x Overpass limit) and
# hangs for hours. Instead we use GRP road density from GEE as a fast proxy,
# then convert to approximate km distance.
print("\n[5/5] Road proximity (GEE — fast method)...")

try:
    # Use Global Roads Open Access Data Set (gROADS) via GEE
    # This runs server-side — no large local downloads
    roads_img = (ee.Image("WWF/HydroSHEDS/03VFDEM")   # placeholder check
                 .unmask(0))

    # Primary: use GRIP global roads rasterised distance
    # We compute distance-to-road using GEE's fastDistanceTransform
    road_fc = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/GRIP4_region5") \
                .filterBounds(study_area)
    road_img    = road_fc.reduceToImage(['GP_RTP'], ee.Reducer.first()).unmask(0).gt(0)
    dist_img    = road_img.fastDistanceTransform(256).sqrt() \
                          .multiply(ee.Image.pixelArea().sqrt()) \
                          .divide(1000)   # convert metres → km
    dist_img    = dist_img.rename('road_proximity_km')
    df_road     = export_and_download(dist_img, 'wf_road', ['road_proximity_km'], scale=1000)
    df          = nearest_merge(df, df_road, 'road_proximity_km')
    df          = df.dropna(subset=['road_proximity_km']).reset_index(drop=True)
    print(f"  {df['road_proximity_km'].min():.2f}-{df['road_proximity_km'].max():.2f} km ✅")

except Exception as e:
    print(f"  GEE road method failed ({e})\n  Using OSMnx tile-by-tile fallback...")

    import math

    def roads_for_tile(lat_min, lat_max, lon_min, lon_max):
        """Download roads for a small tile and return GeoDataFrame."""
        try:
            G = ox.graph_from_bbox(
                north=lat_max, south=lat_min, east=lon_max, west=lon_min,
                network_type='drive', retain_all=False,
                truncate_by_edge=True
            )
            return ox.graph_to_gdfs(G, nodes=False)
        except Exception:
            return None

    # Split study area into 0.2°×0.2° tiles (~22 km each — well within Overpass limit)
    TILE = 0.2
    lat_tiles = [round(LAT_MIN + i * TILE, 4) for i in range(math.ceil((LAT_MAX - LAT_MIN) / TILE))]
    lon_tiles = [round(LON_MIN + i * TILE, 4) for i in range(math.ceil((LON_MAX - LON_MIN) / TILE))]

    all_roads = []
    total = len(lat_tiles) * len(lon_tiles)
    done  = 0
    for lat0 in lat_tiles:
        for lon0 in lon_tiles:
            gdf = roads_for_tile(lat0, min(lat0+TILE, LAT_MAX),
                                  lon0, min(lon0+TILE, LON_MAX))
            if gdf is not None and len(gdf) > 0:
                all_roads.append(gdf[['geometry']])
            done += 1
            print(f"  Tile {done}/{total} done", end='\r', flush=True)

    if all_roads:
        roads_gdf   = gpd.GeoDataFrame(pd.concat(all_roads, ignore_index=True), crs='EPSG:4326')
        roads_union = unary_union(roads_gdf.geometry)
        pts_gs      = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        )
        df['road_proximity_km'] = pts_gs.distance(roads_union) * 111.0
        print(f"\n  {df['road_proximity_km'].min():.2f}-{df['road_proximity_km'].max():.2f} km ✅")
    else:
        print("\n  No road data retrieved — using uniform fallback value of 1.0 km")
        df['road_proximity_km'] = 1.0


# ── 12. FIRE LABELS ───────────────────────────────────────────
print("\nFire labels (NASA FIRMS)...")
try:
    firms_img = (ee.ImageCollection("FIRMS")
                 .filterDate('2022-01-01', '2022-12-31')
                 .filterBounds(study_area)
                 .select('T21').max().gt(330).clip(study_area))
    df_fire = export_and_download(firms_img, 'wf_fire', ['T21'])
    df      = nearest_merge(df, df_fire, 'T21')
    df['fire'] = (df['T21'] > 0).astype(int)
    df.drop(columns=['T21'], inplace=True)
    label_src = "NASA FIRMS"
except Exception as e:
    print(f"  FIRMS failed ({e}) -> rule-based fallback")
    df['fire'] = (
        (df['NDVI']              < 0.25) &
        (df['LST']               > 42.0) &
        (df['rainfall']          < 1.0)  &
        (df['road_proximity_km'] < 5.0)
    ).astype(int)
    label_src = "Rule-based (fallback)"

print(f"  Source: {label_src} | {df['fire'].value_counts().to_dict()}")


# ── 13. FINALISE ──────────────────────────────────────────────
COLS = ['latitude','longitude','NDVI','LST','rainfall',
        'population_density','road_proximity_km','fire']

df_final = df[COLS].dropna().reset_index(drop=True)
df_final['NDVI']               = df_final['NDVI'].clip(-1, 1)
df_final['LST']                = df_final['LST'].clip(0, 80)
df_final['rainfall']           = df_final['rainfall'].clip(0)
df_final['population_density'] = df_final['population_density'].clip(0)
df_final['road_proximity_km']  = df_final['road_proximity_km'].clip(0.01)

print(f"\nShape: {df_final.shape} | Fire%: {df_final['fire'].mean()*100:.1f}%")
print(df_final.describe().round(3))


# ── 14. SAVE ──────────────────────────────────────────────────
df_final.to_csv('wildfire_mysuru_2022.csv', index=False)
print("\nSaved -> wildfire_mysuru_2022.csv ✅")
print("Next: python phase2_temporal.py")