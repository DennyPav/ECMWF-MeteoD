#!/bin/env python3

import os
import json
import sys
import shutil
import zipfile
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import boto3
from collections import Counter
from datetime import datetime, timedelta, timezone
from herbie import Herbie, FastHerbie
from timezonefinder import TimezoneFinder
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# --- CLOUDFLARE R2 ---
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_ENDPOINT   = os.environ.get("R2_ENDPOINT")
R2_BUCKET_NAME = "json-meteod"

# --- CAMS (COPERNICUS) ---
raw_key = os.environ.get("CDS_API_KEY", "")
if ":" in raw_key:
    CDS_KEY = raw_key.split(":", 1)[1]
else:
    CDS_KEY = raw_key
ADS_URL   = "https://ads.atmosphere.copernicus.eu/api"
CAMS_AREA = [48, 6, 35, 19]  # Solo Italia

# --- FILE INPUT ---
FILE_COMUNI_ITALIA = "comuni_italia_all.json"
FILE_COMUNI_ESTERO = "comuni_estero.json"

WORKDIR        = os.getcwd()
VENUES_ITALIA  = os.path.join(WORKDIR, FILE_COMUNI_ITALIA)
VENUES_ESTERO  = os.path.join(WORKDIR, FILE_COMUNI_ESTERO)

# --- GFS (Google Cloud) ---
# Herbie scarica da gs://noaa-public/... automaticamente quando priority=["google"]
# Modello GFS 0.25° - compatibile 1:1 con ECMWF IFS 0.25°
GFS_MODEL   = "gfs"
GFS_PRODUCT = "pgrb2.0p25"   # 0.25° resolution, equivalente a ECMWF 0p25
GFS_SOURCE  = ["google"]     # Herbie cerca SOLO su Google Cloud Storage

# Lapse Rates
LAPSE_DRY   = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P     = 0.012
G  = 9.80665
RD = 287.05

# SOGLIE STAGIONALI
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1,   "end_day": 80,  "fog_rh": 96, "haze_rh": 85, "fog_wind": 7.0,  "haze_wind": 12.0, "fog_max_t": 15.0},
    "spring": {"start_day": 81,  "end_day": 172, "fog_rh": 97, "haze_rh": 85, "fog_wind": 6.0,  "haze_wind": 10.0, "fog_max_t": 20.0},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 98, "haze_rh": 90, "fog_wind": 4.0,  "haze_wind": 9.0,  "fog_max_t": 26.0},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 95, "haze_rh": 88, "fog_wind": 7.0,  "haze_wind": 11.0, "fog_max_t": 20.0}
}

# ============================================================================
# UTILITY
# ============================================================================

def get_r2_client():
    if not R2_ACCESS_KEY or "INSERISCI_QUI" in R2_ACCESS_KEY:
        print("ATTENZIONE: Credenziali R2 non impostate.")
        return None
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto'
    )

def upload_to_r2(s3_client, local_file_path, run_date, run_hour, filename_override=None):
    if not s3_client:
        return False
    try:
        folder_name   = f"{run_date}{run_hour}"
        filename_only = filename_override if filename_override else os.path.basename(local_file_path)
        object_key    = f"GFS/{folder_name}/{filename_only}"   # <-- cartella GFS invece di ECMWF
        s3_client.upload_file(
            local_file_path, R2_BUCKET_NAME, object_key,
            ExtraArgs={'ContentType': 'application/json', 'CacheControl': 'public, max-age=3600'}
        )
        return True
    except Exception as e:
        print(f"[R2] ❌ Errore upload {local_file_path}: {e}", flush=True)
        return False

def get_run_datetime_now_utc():
    """
    GFS gira 4 volte/die: 00Z, 06Z, 12Z, 18Z.
    Diventiamo conservativi come prima: sotto le 6Z usiamo run 18Z del giorno prima,
    tra le 6Z e le 18Z usiamo 00Z, sopra le 18Z usiamo 12Z.
    (Dati GFS disponibili su GCS ~4h dopo il run.)
    """
    now = datetime.now(timezone.utc)
    if now.hour < 6:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "18"
    elif now.hour < 18:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# ---------------------- FUNZIONI METEO ORIGINALI (INTATTE) ----------------------

def get_local_time(dt_utc, lat, lon, tf_instance):
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    timezone_str = tf_instance.timezone_at(lng=lon, lat=lat)
    if timezone_str is None:
        return dt_utc
    try:
        local_tz = ZoneInfo(timezone_str)
        return dt_utc.astimezone(local_tz)
    except Exception:
        return dt_utc

def wet_bulb_celsius(t_c, rh_percent):
    tw = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) \
         + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) \
         + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) \
         - 4.686035
    return tw

def kelvin_to_celsius(k):  return k - 273.15
def mps_to_kmh(mps):        return mps * 3.6

def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67 * t_c)  / (t_c  + 243.5))
    e  = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip(100 * e / es, 0, 100)

def wind_speed_direction(u, v):
    speed_ms = np.sqrt(u**2 + v**2)
    deg      = (np.degrees(np.arctan2(-u, -v)) % 360)
    return speed_ms, deg

def wind_dir_to_cardinal(deg):
    return ['N','NE','E','SE','S','SW','W','NW'][int((deg + 22.5) % 360 // 45)]

def get_season_precise(dt_utc):
    day_of_year = dt_utc.timetuple().tm_yday
    for season, thresh in SEASON_THRESHOLDS.items():
        if thresh["start_day"] <= day_of_year <= thresh["end_day"]:
            return season, thresh
    return "winter", SEASON_THRESHOLDS["winter"]

def altitude_correction(t2m, rh, z_model, z_station, pmsl):
    delta_z = z_model - z_station
    w_moist  = np.clip(rh / 100.0, 0, 1)
    lapse_t  = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr   = t2m + lapse_t * delta_z
    T_mean   = t_corr + 273.15
    p_corr   = pmsl * np.exp(-G * z_station / (RD * T_mean))
    return t_corr, p_corr

def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    octas = clct / 100.0 * 8
    if   octas <= 2: cloud_state = "SERENO"
    elif octas <= 4: cloud_state = "POCO NUVOLOSO"
    elif octas <= 6: cloud_state = "NUVOLOSO"
    else:            cloud_state = "COPERTO"

    wet_bulb     = wet_bulb_celsius(t2m, rh2m)
    prec_type_high = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
    prec_type_low  = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"

    if timestep_hours == 3:
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 5.0, 20.0
    else:
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 10.0, 30.0

    if mucape > 400 and tp_rate > 0.5 * timestep_hours:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} TEMPORALE"

    if tp_rate > 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        if   tp_rate >= prec_intensa_min:  prec_intensity = "INTENSA"
        elif tp_rate >= prec_moderata_min: prec_intensity = "MODERATA"
        else:                              prec_intensity = "DEBOLE"
        return f"{cloud_state} {prec_type_high} {prec_intensity}"
    elif 0.5 <= tp_rate <= 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"

    fog_rh  = season_thresh.get("fog_rh",  95)
    fog_wd  = season_thresh.get("fog_wind", 8)
    fog_t   = season_thresh.get("fog_max_t", 18)
    haze_rh = season_thresh.get("haze_rh",  85)
    haze_wd = season_thresh.get("haze_wind", 12)

    if 0.1 <= tp_rate < 0.5:
        if t2m < fog_t and rh2m >= fog_rh  and wind_kmh <= fog_wd:  return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"
    elif tp_rate < 0.1:
        if t2m < fog_t and rh2m >= fog_rh  and wind_kmh <= fog_wd:  return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        return cloud_state

    return cloud_state

# ---------------------- FUNZIONI ARIA (CAMS) ----------------------

def calculate_caqi(row):
    def get_sub_index(val, grids):
        for i in range(len(grids) - 1):
            low_c, low_i   = grids[i]
            high_c, high_i = grids[i + 1]
            if low_c <= val <= high_c:
                return low_i + (val - low_c) * (high_i - low_i) / (high_c - low_c)
        last_c, last_i = grids[-1]
        return last_i + (val - last_c)

    grid_pm10 = [(0,0),(25,25),(50,50),(90,75),(180,100)]
    grid_pm25 = [(0,0),(15,25),(30,50),(55,75),(110,100)]
    grid_no2  = [(0,0),(50,25),(100,50),(200,75),(400,100)]
    grid_o3   = [(0,0),(60,25),(120,50),(180,75),(240,100)]

    idx_pm10 = get_sub_index(row["pm10"], grid_pm10)
    idx_pm25 = get_sub_index(row["pm25"], grid_pm25)
    idx_no2  = get_sub_index(row["no2"],  grid_no2)
    idx_o3   = get_sub_index(row["o3"],   grid_o3)

    final_aqi = int(round(max(idx_pm10, idx_pm25, idx_no2, idx_o3)))

    if   final_aqi < 30:  label = "Ottima"
    elif final_aqi < 50:  label = "Buona"
    elif final_aqi < 70:  label = "Discreta"
    elif final_aqi < 90:  label = "Pessima"
    elif final_aqi < 120: label = "Inquinata"
    else:                 label = "Molto inquinata"
    return final_aqi, label

def clean_cams_data(df):
    out = []
    for dt, row in df.iterrows():
        out.append({
            "d": dt.strftime("%Y%m%d"), "h": dt.strftime("%H"),
            "pm25": int(row["pm25"]), "pm10": int(row["pm10"]),
            "no2":  int(row["no2"]),  "o3":   int(row["o3"])
        })
    return out

# ============================================================================
# MAPPATURA VARIABILI GFS → NOMI xarray equivalenti a ECMWF
# ============================================================================
#
# GFS (GRIB2 su GCS)           →  nome variabile xarray (cfgrib/herbie)
# ─────────────────────────────────────────────────────────────────────
#  TMP:2 m above ground        →  t2m   (K)
#  DPT:2 m above ground        →  d2m   (K)
#  TCDC:entire atmosphere      →  tcc   (0-1 fraction, mult *100 per %)
#  PRMSL:mean sea level        →  msl   (Pa, div /100 per hPa)
#  APCP:surface (acc)          →  tp    (kg/m² = mm, accumul.)
#  CAPE:surface (MUCAPE proxy) →  cape  (J/kg)  ← GFS non ha MUCAPE separato
#  UGRD:10 m above ground      →  u10   (m/s)
#  VGRD:10 m above ground      →  v10   (m/s)
#  HGT:surface (orografia)     →  orog  (m geopotential height)
#
# NOTA: GFS non ha MUCAPE nativo → usiamo CAPE:surface come proxy.
#       La variabile xarray dopo herbie si chiama "cape".
#       Nel processing sotto viene usata come mucape_arr (stessa logica).

# ============================================================================
# DOWNLOAD DATA  —  GFS via Herbie (Google Cloud Storage)
# ============================================================================

def _build_herbie(run_dt_str, fxx):
    """
    Crea un oggetto Herbie per GFS 0.25° da Google Cloud Storage.
    run_dt_str: es. "2024-05-14 00:00"
    fxx: forecast hour (int)
    """
    return Herbie(
        run_dt_str,
        model=GFS_MODEL,
        product=GFS_PRODUCT,
        fxx=fxx,
        priority=GFS_SOURCE,
        save_dir=f"{WORKDIR}/data_temp",
        overwrite=False,
    )

def _download_and_merge(run_dt_str, steps, variables_regex, out_nc_path):
    """
    Scarica subset di variabili per tutti gli step e li concatena in un
    unico NetCDF con dimensione 'step'.
    variables_regex: stringa regex herbie, es. ":(TMP|DPT|TCDC|PRMSL|APCP|CAPE):..."
    """
    if os.path.exists(out_nc_path) and os.path.getsize(out_nc_path) > 1000:
        print(f"  (cache) {os.path.basename(out_nc_path)}")
        return out_nc_path

    ds_list = []
    for fxx in steps:
        try:
            H  = _build_herbie(run_dt_str, fxx)
            ds = H.xarray(variables_regex, remove_grib=False)
            # Herbie può restituire un dict di datasets se ci sono messaggi multipli
            if isinstance(ds, dict):
                ds = xr.merge(list(ds.values()))
            ds = ds.expand_dims("step").assign_coords(step=[fxx])
            ds_list.append(ds)
        except Exception as e:
            print(f"  ⚠️ GFS step={fxx}: {e}")
            continue

    if not ds_list:
        raise RuntimeError(f"Nessun dato GFS scaricato per {out_nc_path}")

    merged = xr.concat(ds_list, dim="step")
    merged.to_netcdf(out_nc_path)
    merged.close()
    return out_nc_path

def download_data_unified(run_date, run_hour):
    """
    Scarica i dati GFS da Google Cloud Storage tramite Herbie.
    Struttura output equivalente a prima (nc_files dict).
    """
    print(f"--- START DOWNLOAD GFS: {run_date}{run_hour} ---")
    base_dir = f"{WORKDIR}/data_temp/{run_date}{run_hour}"
    os.makedirs(base_dir, exist_ok=True)

    # run datetime string per Herbie: "YYYYMMDD HH:00"
    run_dt_str = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]} {run_hour}:00"

    # ── Step triorari: 0–144 ogni 3h ──────────────────────────────────────
    steps_tri = list(range(0, 145, 3))
    # ── Step esaorari: 150–330/318 ogni 6h ────────────────────────────────
    steps_esa = list(range(150, 331, 6)) if run_hour == "00" else list(range(150, 319, 6))

    # ── Regex variabili ────────────────────────────────────────────────────
    # Variabili principali (atmosfera)
    # GFS GRIB2: TMP=temperatura, DPT=dewpoint, TCDC=total cloud cover,
    #            PRMSL=pressione livello mare, APCP=precipitazione acc., CAPE=cape
    regex_main = ":(TMP:2 m above ground|DPT:2 m above ground|TCDC:entire atmosphere|PRMSL:mean sea level|APCP:surface|CAPE:surface):"
    # Vento
    regex_wind = ":(UGRD:10 m above ground|VGRD:10 m above ground):"
    # Orografia (solo step 0)
    regex_orog = ":HGT:surface:"

    nc_files = {}

    print("⬇️ GFS main triorario...")
    nc_files["main_tri"] = _download_and_merge(
        run_dt_str, steps_tri, regex_main,
        f"{base_dir}/gfs_main_tri.nc"
    )

    print("⬇️ GFS wind triorario...")
    nc_files["wind_tri"] = _download_and_merge(
        run_dt_str, steps_tri, regex_wind,
        f"{base_dir}/gfs_wind_tri.nc"
    )

    print("⬇️ GFS orografia (step 0)...")
    nc_files["orog"] = _download_and_merge(
        run_dt_str, [0], regex_orog,
        f"{base_dir}/gfs_orog.nc"
    )

    print("⬇️ GFS main esaorario...")
    nc_files["main_esa"] = _download_and_merge(
        run_dt_str, steps_esa, regex_main,
        f"{base_dir}/gfs_main_esa.nc"
    )

    # ── CAMS (invariato rispetto all'originale) ────────────────────────────
    cams_zip = f"{base_dir}/cams.zip"
    cams_nc  = f"{base_dir}/cams.nc"

    if not os.path.exists(cams_nc):
        if CDS_KEY:
            print("⬇️ CAMS Italia...", flush=True)
            today      = datetime.now(timezone.utc).date()
            leadtimes  = [str(i) for i in range(97)]
            c_client   = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
            req = {
                "variable": ["nitrogen_dioxide", "ozone",
                             "particulate_matter_2.5um", "particulate_matter_10um"],
                "model": ["ensemble"], "level": ["0"],
                "date":  [f"{today}/{today}"],
                "type":  ["forecast"], "time": ["00:00"],
                "leadtime_hour": leadtimes,
                "data_format": "netcdf_zip", "area": CAMS_AREA
            }
            try:
                c_client.retrieve("cams-europe-air-quality-forecasts", req).download(cams_zip)
                cams_extract_dir = f"{base_dir}/cams_extract"
                os.makedirs(cams_extract_dir, exist_ok=True)
                with zipfile.ZipFile(cams_zip, "r") as z:
                    z.extractall(cams_extract_dir)
                found = [f for f in os.listdir(cams_extract_dir) if f.endswith(".nc")][0]
                shutil.move(os.path.join(cams_extract_dir, found), cams_nc)
                shutil.rmtree(cams_extract_dir)
            except Exception as e:
                print(f"⚠️ CAMS fallito: {e}")
                cams_nc = None
        else:
            cams_nc = None

    nc_files["cams"] = cams_nc
    return nc_files

# ============================================================================
# HELPER: normalizza nomi variabili GFS → nomi usati nel processing
# ============================================================================

def _normalize_gfs_dataset(ds, kind="main"):
    """
    Herbie/cfgrib usa nomi GRIB2 nativi che cambiano leggermente rispetto
    ai nomi ECMWF (es. 't2m' vs 'TMP_2maboveground').
    Questa funzione rinomina tutto per rendere il processing identico.

    kind: "main" | "wind" | "orog"
    """
    rename_map = {}

    # Temperatura 2m
    for cand in ("t2m", "TMP_2maboveground", "tmp", "TMP"):
        if cand in ds: rename_map[cand] = "t2m"; break

    # Dewpoint 2m
    for cand in ("d2m", "DPT_2maboveground", "dpt", "DPT"):
        if cand in ds: rename_map[cand] = "d2m"; break

    # Total cloud cover
    for cand in ("tcc", "TCDC_entireatmosphere", "tcdc", "TCDC"):
        if cand in ds: rename_map[cand] = "tcc"; break

    # Pressione MSL
    for cand in ("msl", "PRMSL_meansealevel", "prmsl", "PRMSL"):
        if cand in ds: rename_map[cand] = "msl"; break

    # Precipitazione accumulata
    for cand in ("tp", "APCP_surface", "apcp", "APCP"):
        if cand in ds: rename_map[cand] = "tp"; break

    # CAPE (proxy MUCAPE)
    for cand in ("cape", "CAPE_surface", "Cape", "CAPE"):
        if cand in ds: rename_map[cand] = "mucape"; break

    # Vento U10
    for cand in ("u10", "UGRD_10maboveground", "ugrd", "UGRD"):
        if cand in ds: rename_map[cand] = "u10"; break

    # Vento V10
    for cand in ("v10", "VGRD_10maboveground", "vgrd", "VGRD"):
        if cand in ds: rename_map[cand] = "v10"; break

    # Orografia (geopotential height in m, non in m²/s² come ECMWF)
    for cand in ("orog", "HGT_surface", "gh", "z", "HGT"):
        if cand in ds: rename_map[cand] = "z"; break

    # Coordinate lat/lon (GFS usa 'latitude'/'longitude' come ECMWF)
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    if rename_map:
        ds = ds.rename(rename_map)

    # GFS tcc è in % (0-100), non in frazione (0-1) come ECMWF.
    # Se il max è <= 1.0 → è già in frazione → moltiplica *100
    if "tcc" in ds:
        if float(ds["tcc"].max()) <= 1.05:
            ds["tcc"] = ds["tcc"] * 100.0

    # GFS msl è in Pa (come ECMWF) → lasciamo la divisione /100 nel processing

    # GFS orografia: HGT:surface è già in metri geopotenziali (non m²/s²),
    # quindi NON serve dividere per 9.81 come facevamo con ECMWF.
    # Rinominiamo internamente in "z" e nel processing useremo direttamente il valore.

    return ds

# ============================================================================
# PROCESSING UNIFICATO
# ============================================================================

def process_unified_venues(venues_path, datasets, run_info, s3_client, tf_instance, process_air=False):
    if not os.path.exists(venues_path):
        return
    with open(venues_path, 'r', encoding='utf-8') as f:
        venues_raw = json.load(f)

    venues = {c: {"lat": float(v[0]), "lon": float(v[1]), "elev": float(v[2])} for c, v in venues_raw.items()}
    print(f"\n🚀 Elaborazione Unificata: {os.path.basename(venues_path)} ({len(venues)} città)")

    ref_dt = datetime.strptime(run_info["run_date"] + run_info["run_hour"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    _, season_thresh = get_season_precise(ref_dt)

    ds_main_tri = datasets["main_tri"]
    ds_wind_tri = datasets["wind_tri"]
    ds_orog     = datasets["orog"]
    ds_main_esa = datasets["main_esa"]
    ds_cams     = datasets.get("cams")

    processed = 0

    for city, info in venues.items():
        try:

            # ── Helper interni (identici all'originale) ──────────────────

            def round_to_nearest_3h(hour):
                return (hour // 3) * 3

            def round_to_nearest_6h(hour):
                return (hour // 6) * 6

            def daily_from_records(records):
                daily    = []
                days_map = {}
                for rec in records:
                    days_map.setdefault(rec["d"], []).append(rec)

                for d in sorted(days_map.keys()):
                    recs  = days_map[d]
                    temps = [r["t"] for r in recs]
                    t_min = min(temps)
                    t_max = max(temps)
                    tp_tot = sum(r["p"] for r in recs)

                    weather_list = [r["w"] for r in recs]
                    has_storm = any("TEMPORALE" in w for w in weather_list)
                    has_snow  = any("NEVE"      in w or "NEVISCHIO"   in w for w in weather_list)
                    has_rain  = any("PIOGGIA"   in w or "PIOGGERELLA" in w for w in weather_list)

                    cloud_counter = Counter()
                    for w in weather_list:
                        if   "COPERTO"       in w: cloud_counter["COPERTO"]       += 1
                        elif "NUVOLOSO"      in w: cloud_counter["NUVOLOSO"]      += 1
                        elif "POCO NUVOLOSO" in w: cloud_counter["POCO NUVOLOSO"] += 1
                        elif "SERENO"        in w: cloud_counter["SERENO"]        += 1

                    c_state = cloud_counter.most_common(1)[0][0] if cloud_counter else "SERENO"

                    if has_storm:
                        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
                        weather_str = f"{c_state} TEMPORALE"
                    elif has_snow:
                        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
                        pint = "INTENSA" if tp_tot >= 30 else ("MODERATA" if tp_tot >= 10 else "DEBOLE")
                        weather_str = f"{c_state} NEVE {pint}"
                    elif has_rain:
                        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
                        pint = "INTENSA" if tp_tot >= 30 else ("MODERATA" if tp_tot >= 10 else "DEBOLE")
                        weather_str = f"{c_state} PIOGGIA {pint}"
                    else:
                        weather_str = c_state

                    daily.append({
                        "d": d,
                        "tmin": round(t_min, 1),
                        "tmax": round(t_max, 1),
                        "p":    round(tp_tot, 1),
                        "w":    weather_str
                    })
                return daily

            # ── Indici griglia ────────────────────────────────────────────
            lat_idx     = np.abs(ds_main_tri.latitude - info['lat']).argmin()
            lon_idx     = np.abs(ds_main_tri.longitude - info['lon']).argmin()
            lat_idx_esa = np.abs(ds_main_esa.latitude - info['lat']).argmin()
            lon_idx_esa = np.abs(ds_main_esa.longitude - info['lon']).argmin()

            # ── TRIORARIO ─────────────────────────────────────────────────
            t2m_k_tri    = ds_main_tri["t2m"].isel(latitude=lat_idx, longitude=lon_idx).values
            td2m_k_tri   = ds_main_tri["d2m"].isel(latitude=lat_idx, longitude=lon_idx).values
            tcc_tri      = ds_main_tri["tcc"].isel(latitude=lat_idx, longitude=lon_idx).values
            msl_tri      = ds_main_tri["msl"].isel(latitude=lat_idx, longitude=lon_idx).values / 100
            tp_cum_tri   = ds_main_tri["tp"].isel(latitude=lat_idx, longitude=lon_idx).values
            mucape_tri   = ds_main_tri["mucape"].isel(latitude=lat_idx, longitude=lon_idx).values
            u10_tri      = ds_wind_tri["u10"].isel(latitude=lat_idx, longitude=lon_idx).values
            v10_tri      = ds_wind_tri["v10"].isel(latitude=lat_idx, longitude=lon_idx).values

            # ── Orografia ─────────────────────────────────────────────────
            # GFS HGT:surface è in metri (geopotential height), non in m²/s²
            # → nessuna divisione per 9.81 necessaria
            z_model = ds_orog["z"].isel(latitude=lat_idx, longitude=lon_idx).values
            if z_model.ndim > 0:
                z_model = float(z_model.squeeze())

            rh2m_tri           = relative_humidity(t2m_k_tri, td2m_k_tri)
            t2m_c_tri          = kelvin_to_celsius(t2m_k_tri)
            t2m_corr_tri, pmsl_corr_tri = altitude_correction(
                t2m_c_tri, rh2m_tri, z_model, info['elev'], msl_tri)
            spd_ms_tri, wd_deg_tri = wind_speed_direction(u10_tri, v10_tri)
            spd_kmh_tri        = mps_to_kmh(spd_ms_tri)

            # Precipitazione: GFS APCP è accumulata dall'inizio del run (mm).
            # Calcoliamo i rate differenziali step-by-step (come con tp ECMWF).
            tp_rate_tri = np.diff(tp_cum_tri, prepend=tp_cum_tri[0])
            # GFS APCP è già in mm (kg/m²), non serve * 1000
            tp_rate_tri = np.clip(tp_rate_tri, 0, None)

            tri_groups = {}
            for i in range(len(t2m_corr_tri)):
                dt_utc   = ref_dt + timedelta(hours=i * 3)
                dt_local = get_local_time(dt_utc, info['lat'], info['lon'], tf_instance)

                rounded_hour = round_to_nearest_3h(dt_local.hour)
                slot_date    = dt_local.date()
                if rounded_hour == 0 and dt_local.hour >= 22:
                    slot_date = slot_date + timedelta(days=1)

                key = (slot_date.strftime("%Y%m%d"), f"{rounded_hour:02d}")

                if key not in tri_groups:
                    tri_groups[key] = {"t":[], "r":[], "ct":[], "wk":[], "pr":[], "tp":[], "mucape":[], "wd":[]}

                tri_groups[key]["t"].append(float(t2m_corr_tri[i]))
                tri_groups[key]["r"].append(float(rh2m_tri[i]))
                tri_groups[key]["ct"].append(float(tcc_tri[i]))
                tri_groups[key]["wk"].append(float(spd_kmh_tri[i]))
                tri_groups[key]["pr"].append(float(pmsl_corr_tri[i]))
                tri_groups[key]["tp"].append(float(tp_rate_tri[i]))
                tri_groups[key]["mucape"].append(float(mucape_tri[i]))
                tri_groups[key]["wd"].append(wind_dir_to_cardinal(float(wd_deg_tri[i])))

            final_tri = []
            for d_key, h_key in sorted(tri_groups.keys()):
                g = tri_groups[(d_key, h_key)]
                t_avg    = float(np.mean(g["t"]))
                rh_avg   = float(np.mean(g["r"]))
                ct_avg   = float(np.mean(g["ct"]))
                wk_avg   = float(np.mean(g["wk"]))
                pr_avg   = float(np.mean(g["pr"]))
                tp_sum   = float(np.sum(g["tp"]))
                mcape    = float(np.mean(g["mucape"]))
                dir_mode = Counter(g["wd"]).most_common(1)[0][0]

                w3 = classify_weather(t_avg, rh_avg, ct_avg, tp_sum, wk_avg, mcape, season_thresh, 3)
                final_tri.append({
                    "d": d_key, "h": h_key,
                    "t": round(t_avg, 1), "r": round(rh_avg),
                    "p": round(tp_sum, 1), "pr": round(pr_avg),
                    "v": round(wk_avg, 1), "vd": dir_mode, "w": w3
                })

            daily_tri = daily_from_records(final_tri)

            # ── ESAORARIO ─────────────────────────────────────────────────
            t2m_k_esa  = ds_main_esa["t2m"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            td2m_k_esa = ds_main_esa["d2m"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            tcc_esa    = ds_main_esa["tcc"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            msl_esa    = ds_main_esa["msl"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values / 100
            tp_cum_esa = ds_main_esa["tp"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            mucape_esa = ds_main_esa["mucape"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values

            rh2m_esa   = relative_humidity(t2m_k_esa, td2m_k_esa)
            t2m_c_esa  = kelvin_to_celsius(t2m_k_esa)
            t2m_corr_esa, pmsl_corr_esa = altitude_correction(
                t2m_c_esa, rh2m_esa, z_model, info['elev'], msl_esa)
            tp_rate_esa = np.clip(np.diff(tp_cum_esa, prepend=tp_cum_esa[0]), 0, None)

            esa_groups = {}
            for i in range(len(t2m_corr_esa)):
                dt_utc   = ref_dt + timedelta(hours=150 + i * 6)
                dt_local = get_local_time(dt_utc, info['lat'], info['lon'], tf_instance)

                rounded_hour = round_to_nearest_6h(dt_local.hour)
                slot_date    = dt_local.date()
                if rounded_hour == 0 and dt_local.hour >= 20:
                    slot_date = slot_date + timedelta(days=1)

                key = (slot_date.strftime("%Y%m%d"), f"{rounded_hour:02d}")

                if key not in esa_groups:
                    esa_groups[key] = {"t":[], "r":[], "ct":[], "pr":[], "tp":[], "mucape":[]}

                esa_groups[key]["t"].append(float(t2m_corr_esa[i]))
                esa_groups[key]["r"].append(float(rh2m_esa[i]))
                esa_groups[key]["ct"].append(float(tcc_esa[i]))
                esa_groups[key]["pr"].append(float(pmsl_corr_esa[i]))
                esa_groups[key]["tp"].append(float(tp_rate_esa[i]))
                esa_groups[key]["mucape"].append(float(mucape_esa[i]))

            final_esa = []
            for d_key, h_key in sorted(esa_groups.keys()):
                g      = esa_groups[(d_key, h_key)]
                t_avg  = float(np.mean(g["t"]))
                rh_avg = float(np.mean(g["r"]))
                ct_avg = float(np.mean(g["ct"]))
                pr_avg = float(np.mean(g["pr"]))
                tp_sum = float(np.sum(g["tp"]))
                mcape  = float(np.mean(g["mucape"]))

                w6 = classify_weather(t_avg, rh_avg, ct_avg, tp_sum, 5.0, mcape, season_thresh, 6)
                final_esa.append({
                    "d": d_key, "h": h_key,
                    "t": round(t_avg, 1), "r": round(rh_avg),
                    "p": round(tp_sum, 1), "pr": round(pr_avg),
                    "v": None, "vd": None, "w": w6
                })

            daily_esa = daily_from_records(final_esa)

            # ── MERGE GIORNALIERO (identico all'originale) ─────────────────
            final_daily = list(daily_tri)
            if daily_tri and daily_esa:
                last_tri  = daily_tri[-1]
                first_esa = daily_esa[0]

                if last_tri["d"] == first_esa["d"]:
                    w_final = first_esa["w"]
                    if any(k in last_tri["w"]  for k in ("PIOGGIA","TEMPORALE","NEVE")): w_final = last_tri["w"]
                    elif any(k in first_esa["w"] for k in ("PIOGGIA","TEMPORALE","NEVE")): w_final = first_esa["w"]

                    merged_day = {
                        "d":    last_tri["d"],
                        "tmin": min(last_tri["tmin"], first_esa["tmin"]),
                        "tmax": max(last_tri["tmax"], first_esa["tmax"]),
                        "p":    round(last_tri["p"] + first_esa["p"], 1),
                        "w":    w_final
                    }
                    final_daily[-1] = merged_day
                    final_daily.extend(daily_esa[1:])
                else:
                    final_daily.extend(daily_esa)
            else:
                final_daily.extend(daily_esa)

            # ── Payload ───────────────────────────────────────────────────
            city_data = {
                "r":          run_info["run_str"],
                "c":          city,
                "x":          info['lat'],
                "y":          info['lon'],
                "z":          info['elev'],
                "TRIORARIO":  final_tri,
                "ESAORARIO":  final_esa,
                "GIORNALIERO": final_daily
            }

            # ── ARIA (CAMS) — identico all'originale ──────────────────────
            if process_air and ds_cams is not None:
                try:
                    sel_c = ds_cams.sel(lat=info['lat'], lon=info['lon'], method="nearest")
                    df    = sel_c.to_dataframe().reset_index()

                    base_utc = pd.Timestamp.now(timezone.utc).normalize()
                    if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]) and df["time"].nunique() > 1:
                        df["t"] = df["time"]
                    elif "leadtime" in df.columns:
                        df["t"] = base_utc + pd.to_timedelta(df["leadtime"])
                    elif "step" in df.columns:
                        df["t"] = base_utc + df["step"]
                    else:
                        df["t"] = [base_utc + pd.Timedelta(hours=h) for h in range(len(df))]

                    df = df.set_index("t").sort_index()
                    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
                    df        = df.tz_convert("Europe/Rome")
                    cutoff_dt = df.index[0] + pd.Timedelta(days=4)
                    df        = df[df.index < cutoff_dt]
                    df        = df[["pm25", "pm10", "no2", "o3"]]

                    aria_h      = clean_cams_data(df)
                    first_hour  = df.index[0].hour
                    offset_str  = f"{first_hour}h"
                    aria_3h     = clean_cams_data(df.resample("3h", offset=offset_str).mean().dropna())

                    df_d    = df.resample("1D").mean()
                    counts  = df.resample("1D").count()["pm25"]
                    df_d    = df_d[counts >= 18].round(0).astype(int)

                    aria_d  = []
                    for dt, row in df_d.iterrows():
                        val, lbl = calculate_caqi(row)
                        aria_d.append({
                            "d": dt.strftime("%Y%m%d"),
                            "pm25": int(row["pm25"]), "pm10": int(row["pm10"]),
                            "no2":  int(row["no2"]),  "o3":   int(row["o3"]),
                            "aqi_value": val, "aqi_class": lbl
                        })

                    city_data["ARIA_ORARIO"]    = aria_h
                    city_data["ARIA_TRIORARIO"] = aria_3h
                    city_data["ARIA_GIORNO"]    = aria_d
                except Exception:
                    pass

            # ── Salvataggio & Upload ──────────────────────────────────────
            safe_name = city.replace("'", " ").replace("/", "-")
            out_file  = os.path.join(run_info["outdir"], f"{safe_name}_gfs.json")  # _gfs invece di _ecmwf

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(city_data, f, separators=(",", ":"), ensure_ascii=False)

            if s3_client:
                upload_to_r2(s3_client, out_file, run_info["run_date"], run_info["run_hour"], os.path.basename(out_file))

            processed += 1
            if processed % 50 == 0:
                print(f"{processed}...", end=" ", flush=True)

        except Exception as e:
            print(f"\n⚠️ Errore {city}: {e}")
            continue

    print(f"\n✅ Completato {os.path.basename(venues_path)}: {processed} città.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    rd, rh  = get_run_datetime_now_utc()
    run_str = f"{rd}{rh}"
    print(f"--- RUN GFS: {run_str} ---")

    # 1. Download
    files = download_data_unified(rd, rh)

    # 2. Setup
    tf = TimezoneFinder(in_memory=True)
    s3 = get_r2_client()
    if s3: print("✅ R2 Connesso.")

    outdir = f"{WORKDIR}/{run_str}"
    os.makedirs(outdir, exist_ok=True)
    run_info = {"run_date": rd, "run_hour": rh, "run_str": run_str, "outdir": outdir}

    # 3. Apertura e normalizzazione Dataset
    print("📂 Apertura Dataset GFS (Meteo + Aria)...")
    try:
        ds_main_tri_raw = xr.open_dataset(files["main_tri"])
        ds_wind_tri_raw = xr.open_dataset(files["wind_tri"])
        ds_orog_raw     = xr.open_dataset(files["orog"])
        ds_main_esa_raw = xr.open_dataset(files["main_esa"])

        datasets = {
            "main_tri": _normalize_gfs_dataset(ds_main_tri_raw, "main"),
            "wind_tri": _normalize_gfs_dataset(ds_wind_tri_raw, "wind"),
            "orog":     _normalize_gfs_dataset(ds_orog_raw,     "orog"),
            "main_esa": _normalize_gfs_dataset(ds_main_esa_raw, "main"),
            "cams":     xr.open_dataset(files["cams"]) if files.get("cams") else None
        }

        # Prep CAMS (identico all'originale)
        if datasets["cams"]:
            rn = {}
            for v in datasets["cams"].data_vars:
                if "pm2p5" in v: rn[v] = "pm25"
                elif "pm10" in v: rn[v] = "pm10"
                elif "no2"  in v: rn[v] = "no2"
                elif "o3"   in v: rn[v] = "o3"
            if rn: datasets["cams"] = datasets["cams"].rename(rn)
            if "latitude" in datasets["cams"].coords:
                datasets["cams"] = datasets["cams"].rename({"latitude": "lat", "longitude": "lon"})
            datasets["cams"] = datasets["cams"][["pm25", "pm10", "no2", "o3"]]

    except Exception as e:
        print(f"❌ Errore Dataset: {e}")
        return

    # 4. Processing
    process_unified_venues(VENUES_ITALIA, datasets, run_info, s3, tf, process_air=True)
    process_unified_venues(VENUES_ESTERO, datasets, run_info, s3, tf, process_air=False)

    for d in datasets.values():
        if d: d.close()

    # shutil.rmtree(f"{WORKDIR}/data_temp/{run_str}")
    print("\n🏁 FINITO.")

if __name__ == "__main__":
    main()
