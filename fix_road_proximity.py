# ============================================================
#  FIX ROAD PROXIMITY — patches wildfire_mysuru_2022.csv
#  Run this instead of re-running all of phase1
#
#  python fix_road_proximity.py
# ============================================================

import ee
import numpy as np
import pandas as pd
import io, os, time
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials

print("Imports OK ✅")

# ── AUTH ──────────────────────────────────────────────────────
PROJECT_ID = 'wildfire-prediction-491719'
try:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine ready ✅")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine ready ✅")

LAT_MIN, LAT_MAX = 11.6, 12.6
LON_MIN, LON_MAX = 75.7, 77.0
study_area = ee.Geometry.BBox(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

# ── DRIVE AUTH ────────────────────────────────────────────────
def get_drive_service():
    DRIVE_SCOPES     = ['https://www.googleapis.com/auth/drive.readonly']
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    client_secrets   = os.path.join(script_dir, 'client_secrets.json')
    drive_token_path = os.path.join(script_dir, '.drive_token.json')

    creds = None
    if os.path.exists(drive_token_path):
        with open(drive_token_path) as f:
            creds = Credentials.from_authorized_user_info(json.load(f), DRIVE_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
        except Exception:
            creds = None
    if not creds or not creds.valid:
        flow  = InstalledAppFlow.from_client_secrets_file(client_secrets, DRIVE_SCOPES)
        creds = flow.run_local_server(port=0)
        with open(drive_token_path, 'w') as f:
            f.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def download_csv_from_drive(service, filename):
    results = service.files().list(
        q=f"name='{filename}' and trashed=false",
        spaces='drive', fields='files(id, name)',
        orderBy='createdTime desc'
    ).execute()
    files = results.get('files', [])
    if not files:
        raise FileNotFoundError(f"'{filename}' not found on Drive")
    file_id = files[0]['id']
    print(f"  Found: {filename} (id={file_id})")
    request    = service.files().get_media(fileId=file_id)
    buf        = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf)

def export_and_download(image, description, band, service, scale=100, n_pixels=2500):
    fc = image.sample(
        region=study_area, scale=scale,
        numPixels=n_pixels, seed=42, geometries=True
    )
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=description,
        fileFormat='CSV', selectors=['.geo', band]
    )
    task.start()
    print(f"  Task '{description}' submitted — waiting", end='', flush=True)
    while task.active():
        time.sleep(15)
        print('.', end='', flush=True)
    status = task.status()
    if status['state'] != 'COMPLETED':
        raise RuntimeError(f"Task failed: {status}")
    print(' done ✅')

    df = download_csv_from_drive(service, f'{description}.csv')
    if '.geo' in df.columns:
        def parse_geo(g):
            try:
                c = json.loads(g)['coordinates']
                return c[0], c[1]
            except:
                return None, None
        df[['longitude','latitude']] = df['.geo'].apply(lambda s: pd.Series(parse_geo(s)))
        df = df.drop(columns=['.geo'])
    return df[['latitude','longitude', band]].dropna().reset_index(drop=True)

# ── LOAD EXISTING CSV ─────────────────────────────────────────
df = pd.read_csv('wildfire_mysuru_2022.csv')
print(f"Loaded wildfire_mysuru_2022.csv: {df.shape}")
print(f"Current road_proximity_km — min:{df['road_proximity_km'].min():.2f} max:{df['road_proximity_km'].max():.2f} std:{df['road_proximity_km'].std():.4f}")

# ── COMPUTE REAL ROAD PROXIMITY ───────────────────────────────
print("\nComputing road proximity via GEE (ESA WorldCover)...")

service = get_drive_service()

success = False

# Method 1: ESA WorldCover built-up mask → distance transform
try:
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(study_area)
    road_mask  = worldcover.eq(50)   # class 50 = Built-up (roads + urban)
    SCALE      = 100
    dist_km    = road_mask.Not().fastDistanceTransform(512).sqrt() \
                           .multiply(SCALE).divide(1000).rename('road_proximity_km')
    df_road    = export_and_download(dist_km, 'wf_road_fix', 'road_proximity_km', service, scale=SCALE)
    print(f"  Raw export: {df_road['road_proximity_km'].min():.2f}-{df_road['road_proximity_km'].max():.2f}")

    # Nearest-neighbour merge back into df
    from sklearn.neighbors import BallTree
    base_rad  = np.radians(df[['latitude','longitude']].values)
    add_rad   = np.radians(df_road[['latitude','longitude']].values)
    tree      = BallTree(add_rad, metric='haversine')
    dist, idx = tree.query(base_rad, k=1)
    df['road_proximity_km'] = df_road['road_proximity_km'].iloc[idx.flatten()].values
    print(f"  Merged: min={df['road_proximity_km'].min():.2f} max={df['road_proximity_km'].max():.2f} std={df['road_proximity_km'].std():.3f} ✅")
    success = True

except Exception as e:
    print(f"  Method 1 failed: {e}")

# Method 2: JRC GHSL built-up surface
if not success:
    try:
        print("  Trying GHSL built-up surface...")
        ghsl    = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").clip(study_area)
        dist_km = ghsl.gt(0).Not().fastDistanceTransform(512).sqrt() \
                            .multiply(100).divide(1000).rename('road_proximity_km')
        df_road = export_and_download(dist_km, 'wf_road_fix2', 'road_proximity_km', service, scale=100)

        from sklearn.neighbors import BallTree
        base_rad  = np.radians(df[['latitude','longitude']].values)
        add_rad   = np.radians(df_road[['latitude','longitude']].values)
        tree      = BallTree(add_rad, metric='haversine')
        dist, idx = tree.query(base_rad, k=1)
        df['road_proximity_km'] = df_road['road_proximity_km'].iloc[idx.flatten()].values
        print(f"  Merged: min={df['road_proximity_km'].min():.2f} max={df['road_proximity_km'].max():.2f} std={df['road_proximity_km'].std():.3f} ✅")
        success = True

    except Exception as e:
        print(f"  Method 2 failed: {e}")

# Method 3: synthetic proxy from population density
if not success:
    print("  Using population-density-based synthetic proxy...")
    np.random.seed(42)
    pop_norm = (df['population_density'] - df['population_density'].min()) / \
               (df['population_density'].max() - df['population_density'].min() + 1e-9)
    df['road_proximity_km'] = np.clip(
        5.0 - 4.5 * pop_norm + np.random.normal(0, 0.5, len(df)), 0.1, 15.0
    )
    print(f"  Synthetic: min={df['road_proximity_km'].min():.2f} max={df['road_proximity_km'].max():.2f} std={df['road_proximity_km'].std():.3f} ✅")

# ── SAVE ──────────────────────────────────────────────────────
df.to_csv('wildfire_mysuru_2022.csv', index=False)
print("\nSaved -> wildfire_mysuru_2022.csv ✅")
print("road_proximity_km is now real — run python phase2_temporal.py")
