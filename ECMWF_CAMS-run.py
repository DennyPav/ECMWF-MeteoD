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
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client
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
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_BUCKET_NAME = "json-meteod"

# --- CAMS (COPERNICUS) ---
raw_key = os.environ.get("CDS_API_KEY", "") 
if ":" in raw_key:
    CDS_KEY = raw_key.split(":", 1)[1]
else:
    CDS_KEY = raw_key
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
CAMS_AREA = [48, 6, 35, 19] # Solo Italia

# --- FILE INPUT ---
FILE_COMUNI_ITALIA = "comuni_italia_all.json"
FILE_COMUNI_ESTERO = "comuni_estero.json"

WORKDIR = os.getcwd()
VENUES_ITALIA = os.path.join(WORKDIR, FILE_COMUNI_ITALIA)
VENUES_ESTERO = os.path.join(WORKDIR, FILE_COMUNI_ESTERO)

# Lapse Rates
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.012
G = 9.80665
RD = 287.05

# SOGLIE STAGIONALI (TUE ORIGINALI)
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 96, "haze_rh": 85, "fog_wind": 7.0, "haze_wind": 12.0, "fog_max_t": 15.0},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 97, "haze_rh": 85, "fog_wind": 6.0, "haze_wind": 10.0, "fog_max_t": 20.0},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 98, "haze_rh": 90, "fog_wind": 4.0, "haze_wind": 9.0, "fog_max_t": 26.0},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 95, "haze_rh": 88, "fog_wind": 7.0, "haze_wind": 11.0, "fog_max_t": 20.0}
}

# ============================================================================
# UTILITY
# ============================================================================

def get_r2_client():
    if "INSERISCI_QUI" in R2_ACCESS_KEY or not R2_ACCESS_KEY:
        print("ATTENZIONE: Credenziali R2 non impostate.")
        return None
    return boto3.client('s3', endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY, region_name='auto')

def upload_to_r2(s3_client, local_file_path, run_date, run_hour, filename_override=None):
    if not s3_client: return False
    try:
        folder_name = f"{run_date}{run_hour}"
        filename_only = filename_override if filename_override else os.path.basename(local_file_path)
        object_key = f"ECMWF/{folder_name}/{filename_only}"
        s3_client.upload_file(
            local_file_path, R2_BUCKET_NAME, object_key,
            ExtraArgs={'ContentType': 'application/json', 'CacheControl': 'public, max-age=3600'}
        )
        return True
    except Exception as e:
        print(f"[R2] ❌ Errore upload {local_file_path}: {e}", flush=True)
        return False

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 6: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 18: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# ---------------------- TUE FUNZIONI METEO ORIGINALI (INTATTE) ----------------------

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

def convert_grib_to_nc_global(infile):
    # Se esiste già non lo rifacciamo (ottimizzazione)
    if os.path.exists(infile.replace(".grib", ".nc")): return infile.replace(".grib", ".nc")
    print(f"Conversione GRIB -> NC: {infile} ...")
    ds = xr.open_dataset(infile, engine="cfgrib")
    outfile = infile.replace(".grib", ".nc")
    ds.to_netcdf(outfile) 
    ds.close()
    return outfile

def kelvin_to_celsius(k): return k - 273.15
def mps_to_kmh(mps): return mps * 3.6

def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip(100 * e / es, 0, 100)

def wind_speed_direction(u, v):
    speed_ms = np.sqrt(u**2 + v**2)
    deg = (np.degrees(np.arctan2(-u, -v)) % 360)
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
    w_moist = np.clip(rh / 100.0, 0, 1)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr = t2m + lapse_t * delta_z
    T_mean = t_corr + 273.15
    p_corr = pmsl * np.exp(-G * z_station / (RD * T_mean))
    return t_corr, p_corr

def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    octas = clct / 100.0 * 8
    if octas <= 2: cloud_state = "SERENO"
    elif octas <= 4: cloud_state = "POCO NUVOLOSO"
    elif octas <= 6: cloud_state = "NUVOLOSO"
    else: cloud_state = "COPERTO"

    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    prec_type_high = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
    prec_type_low = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"

    if timestep_hours == 3:
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 5.0, 20.0
    else:
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 10.0, 30.0

    if mucape > 400 and tp_rate > 0.5 * timestep_hours:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} TEMPORALE"

    if tp_rate > 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        if tp_rate >= prec_intensa_min: prec_intensity = "INTENSA"
        elif tp_rate >= prec_moderata_min: prec_intensity = "MODERATA"
        else: prec_intensity = "DEBOLE"
        return f"{cloud_state} {prec_type_high} {prec_intensity}"
    elif 0.5 <= tp_rate <= 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"
    
    fog_rh = season_thresh.get("fog_rh", 95)
    fog_wd = season_thresh.get("fog_wind", 8)
    fog_t = season_thresh.get("fog_max_t", 18)
    haze_rh = season_thresh.get("haze_rh", 85)
    haze_wd = season_thresh.get("haze_wind", 12)

    if 0.1 <= tp_rate < 0.5:
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd: return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"
    elif tp_rate < 0.1:
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd: return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        return cloud_state
    
    return cloud_state

def calculate_daily_summaries(records, clct_arr, tp_arr, mucape_arr, season_thresh, timestep_hours):
    daily = []
    days_map = {}
    for i, rec in enumerate(records):
        days_map.setdefault(rec["d"], []).append((i, rec))
    
    for d, items in days_map.items():
        idxs = [x[0] for x in items]
        recs = [x[1] for x in items]
        temps = [r["t"] for r in recs]
        t_min, t_max = min(temps), max(temps)
        tp_tot = sum([r["p"] for r in recs])
        
        snow_steps = 0
        rain_steps = 0
        has_storm = False
        has_significant_snow_or_rain = False
        
        for r in recs:
            wtxt = r.get("w", "")
            if "TEMPORALE" in wtxt: has_storm = True
            if "PIOGGIA" in wtxt or "NEVE" in wtxt: has_significant_snow_or_rain = True
            wb = wet_bulb_celsius(r["t"], r["r"])
            if wb < 0.5: snow_steps += 1
            else: rain_steps += 1
        
        is_snow_day = snow_steps > rain_steps
        clct_mean = np.mean(clct_arr[idxs])
        octas = clct_mean / 100.0 * 8
        
        if octas <= 2: c_state = "SERENO"
        elif octas <= 4: c_state = "POCO NUVOLOSO"
        elif octas <= 6: c_state = "NUVOLOSO"
        else: c_state = "COPERTO"
        
        weather_str = c_state
        if has_storm:
            if c_state == "SERENO": c_state = "POCO NUVOLOSO"
            weather_str = f"{c_state} TEMPORALE"
        elif has_significant_snow_or_rain:
            ptype = "NEVE" if is_snow_day else "PIOGGIA"
            if tp_tot >= 30: pint = "INTENSA"
            elif tp_tot >= 10: pint = "MODERATA"
            else: pint = "DEBOLE"
            if c_state == "SERENO": c_state = "POCO NUVOLOSO"
            weather_str = f"{c_state} {ptype} {pint}"
            
        daily.append({
            "d": d, "tmin": round(t_min, 1), "tmax": round(t_max, 1),
            "p": round(tp_tot, 1), "w": weather_str
        })
    return daily

# ---------------------- FUNZIONI ARIA (CAMS) ----------------------

def calculate_caqi(row):
    def get_sub_index(val, grids):
        for i in range(len(grids)-1):
            low_c, low_i = grids[i]
            high_c, high_i = grids[i+1]
            if low_c <= val <= high_c:
                return low_i + (val - low_c) * (high_i - low_i) / (high_c - low_c)
        last_c, last_i = grids[-1]
        return last_i + (val - last_c)
    
    grid_pm10 = [(0,0), (25,25), (50,50), (90,75), (180,100)]
    grid_pm25 = [(0,0), (15,25), (30,50), (55,75), (110,100)]
    grid_no2  = [(0,0), (50,25), (100,50), (200,75), (400,100)]
    grid_o3   = [(0,0), (60,25), (120,50), (180,75), (240,100)]

    idx_pm10 = get_sub_index(row["pm10"], grid_pm10)
    idx_pm25 = get_sub_index(row["pm25"], grid_pm25)
    idx_no2 = get_sub_index(row["no2"], grid_no2)
    idx_o3 = get_sub_index(row["o3"], grid_o3)

    final_aqi = int(round(max(idx_pm10, idx_pm25, idx_no2, idx_o3)))

    if final_aqi < 25: label = "Molto Basso"
    elif final_aqi < 50: label = "Basso"
    elif final_aqi < 75: label = "Medio"
    elif final_aqi < 100: label = "Alto"
    else: label = "Molto Alto"
    return final_aqi, label

def clean_cams_data(df):
    out = []
    for dt, row in df.iterrows():
        out.append({
            "d": dt.strftime("%Y%m%d"), "h": dt.strftime("%H"),
            "pm25": int(row["pm25"]), "pm10": int(row["pm10"]),
            "no2": int(row["no2"]), "o3": int(row["o3"])
        })
    return out

# ============================================================================
# DOWNLOAD DATA
# ============================================================================

def download_data_unified(run_date, run_hour):
    print(f"--- START DOWNLOAD: {run_date}{run_hour} ---")
    base_dir = f"{WORKDIR}/data_temp/{run_date}{run_hour}"
    os.makedirs(base_dir, exist_ok=True)
    
    # --- 1. ECMWF ---
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    
    # Triorario
    steps_tri = list(range(0, 145, 3))
    files = {
        "main_tri": (f"{base_dir}/ecmwf_main_tri.grib", ["2t", "2d", "tcc", "msl", "tp", "mucape"], steps_tri),
        "wind_tri": (f"{base_dir}/ecmwf_wind_tri.grib", ["10u", "10v"], steps_tri),
        "orog": (f"{base_dir}/ecmwf_orog.grib", ["z"], [0])
    }
    # Esaorario
    steps_esa = list(range(150, 331, 6)) if run_hour == "00" else list(range(150, 319, 6))
    files["main_esa"] = (f"{base_dir}/ecmwf_main_esa.grib", ["2t", "2d", "tcc", "msl", "tp", "mucape"], steps_esa)
    
    nc_files = {}
    for key, (path, params, steps) in files.items():
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"⬇️ ECMWF {key}...")
            client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc", step=steps, param=params, target=path)
        nc_files[key] = convert_grib_to_nc_global(path)

    # --- 2. CAMS (Solo se necessario per Italia) ---
    cams_zip = f"{base_dir}/cams.zip"
    cams_nc = f"{base_dir}/cams.nc"

    if not os.path.exists(cams_nc):
        if CDS_KEY:
            print("⬇️ CAMS Italia...", flush=True)
            today = datetime.now(timezone.utc).date()
            leadtimes = [str(i) for i in range(97)]
            c_client = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
            req = {
                "variable": ["nitrogen_dioxide", "ozone", "particulate_matter_2.5um", "particulate_matter_10um"],
                "model": ["ensemble"], "level": ["0"], "date": [f"{today}/{today}"],
                "type": ["forecast"], "time": ["00:00"], "leadtime_hour": leadtimes,
                "data_format": "netcdf_zip", "area": CAMS_AREA
            }
            try:
                c_client.retrieve("cams-europe-air-quality-forecasts", req).download(cams_zip)
            
                # ESTRAI IN SUBDIRECTORY SEPARATA
                cams_extract_dir = f"{base_dir}/cams_extract"
                os.makedirs(cams_extract_dir, exist_ok=True)
            
                with zipfile.ZipFile(cams_zip, "r") as z: 
                    z.extractall(cams_extract_dir)
            
                # CERCA SOLO NELLA SUBDIRECTORY CAMS
                found = [f for f in os.listdir(cams_extract_dir) if f.endswith(".nc")][0]
                shutil.move(os.path.join(cams_extract_dir, found), cams_nc)
            
                # Cleanup subdirectory temporanea
                shutil.rmtree(cams_extract_dir)
            
            except Exception as e:
                print(f"⚠️ CAMS fallito: {e}")
                cams_nc = None
        else:
            cams_nc = None

    nc_files["cams"] = cams_nc
    return nc_files


# ============================================================================
# PROCESSING UNIFICATO (METEO ORIGINAL + CAMS)
# ============================================================================

def process_unified_venues(venues_path, datasets, run_info, s3_client, tf_instance, process_air=False):
    if not os.path.exists(venues_path): return
    with open(venues_path, 'r', encoding='utf-8') as f: venues_raw = json.load(f)
    
    venues = {c: {"lat": float(v[0]), "lon": float(v[1]), "elev": float(v[2])} for c, v in venues_raw.items()}
    print(f"\n🚀 Elaborazione Unificata: {os.path.basename(venues_path)} ({len(venues)} città)")

    ref_dt = datetime.strptime(run_info["run_date"] + run_info["run_hour"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    _, season_thresh = get_season_precise(ref_dt)
    
    # Dataset Meteo
    ds_main_tri = datasets["main_tri"]
    ds_wind_tri = datasets["wind_tri"]
    ds_orog = datasets["orog"]
    ds_main_esa = datasets["main_esa"]
    # Dataset Cams
    ds_cams = datasets.get("cams")

    processed = 0

    for city, info in venues.items():
        try:
            # --- 1. METEO (CODICE ORIGINALE ADATTATO AL LOOP) ---
            # Indici Griglia
            lat_idx = np.abs(ds_main_tri.latitude - info['lat']).argmin()
            lon_idx = np.abs(ds_main_tri.longitude - info['lon']).argmin()
            lat_idx_esa = np.abs(ds_main_esa.latitude - info['lat']).argmin()
            lon_idx_esa = np.abs(ds_main_esa.longitude - info['lon']).argmin()

            # TRIORARIO
            t2m_k = ds_main_tri["t2m"].isel(latitude=lat_idx, longitude=lon_idx).values
            td2m_k = ds_main_tri["d2m"].isel(latitude=lat_idx, longitude=lon_idx).values
            tcc = ds_main_tri["tcc"].isel(latitude=lat_idx, longitude=lon_idx).values * 100
            msl = ds_main_tri["msl"].isel(latitude=lat_idx, longitude=lon_idx).values / 100
            tp_cum = ds_main_tri["tp"].isel(latitude=lat_idx, longitude=lon_idx).values
            mucape = ds_main_tri["mucape"].isel(latitude=lat_idx, longitude=lon_idx).values
            u10 = ds_wind_tri["u10"].isel(latitude=lat_idx, longitude=lon_idx).values
            v10 = ds_wind_tri["v10"].isel(latitude=lat_idx, longitude=lon_idx).values
            z_model = ds_orog["z"].isel(latitude=lat_idx, longitude=lon_idx).values / 9.81

            rh2m = relative_humidity(t2m_k, td2m_k)
            t2m_c = kelvin_to_celsius(t2m_k)
            t2m_corr, pmsl_corr = altitude_correction(t2m_c, rh2m, z_model, info['elev'], msl)
            spd_ms, wd_deg = wind_speed_direction(u10, v10)
            spd_kmh = mps_to_kmh(spd_ms)
            tp_rate = np.diff(tp_cum, prepend=tp_cum[0]) * 1000

            trihourly_data = []
            for i in range(len(t2m_corr)):
                dt_utc = ref_dt + timedelta(hours=i*3)
                dt_local = get_local_time(dt_utc, info['lat'], info['lon'], tf_instance)
                w = classify_weather(t2m_corr[i], rh2m[i], tcc[i], tp_rate[i], spd_kmh[i], mucape[i], season_thresh, 3)
                trihourly_data.append({
                    "d": dt_local.strftime("%Y%m%d"), "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_corr[i]), 1), "r": round(float(rh2m[i])),
                    "p": round(float(tp_rate[i]), 1), "pr": round(float(pmsl_corr[i])),
                    "v": round(float(spd_kmh[i]), 1), "vd": wind_dir_to_cardinal(wd_deg[i]), "w": w
                })
            
            daily_tri = calculate_daily_summaries(trihourly_data, tcc, tp_rate, mucape, season_thresh, 3)

            # ESAORARIO
            t2m_k_e = ds_main_esa["t2m"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            td2m_k_e = ds_main_esa["d2m"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            tcc_e = ds_main_esa["tcc"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values * 100
            msl_e = ds_main_esa["msl"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values / 100
            tp_cum_e = ds_main_esa["tp"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            mucape_e = ds_main_esa["mucape"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            
            rh2m_e = relative_humidity(t2m_k_e, td2m_k_e)
            t2m_c_e = kelvin_to_celsius(t2m_k_e)
            t2m_corr_e, pmsl_corr_e = altitude_correction(t2m_c_e, rh2m_e, z_model, info['elev'], msl_e)
            tp_rate_e = np.diff(tp_cum_e, prepend=tp_cum_e[0]) * 1000

            esaorario_data = []
            for i in range(len(t2m_corr_e)):
                dt_utc = ref_dt + timedelta(hours=150 + i*6)
                dt_local = get_local_time(dt_utc, info['lat'], info['lon'], tf_instance)
                w = classify_weather(t2m_corr_e[i], rh2m_e[i], tcc_e[i], tp_rate_e[i], 5.0, mucape_e[i], season_thresh, 6)
                esaorario_data.append({
                    "d": dt_local.strftime("%Y%m%d"), "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_corr_e[i]), 1), "r": round(float(rh2m_e[i])),
                    "p": round(float(tp_rate_e[i]), 1), "pr": round(float(pmsl_corr_e[i])),
                    "v": None, "vd": None, "w": w
                })

            daily_esa = calculate_daily_summaries(esaorario_data, tcc_e, tp_rate_e, mucape_e, season_thresh, 6)

            # MERGE GIORNALIERO (TUO CODICE ORIGINALE)
            final_daily = list(daily_tri)
            if daily_tri and daily_esa:
                last_tri = daily_tri[-1]
                first_esa = daily_esa[0]
                if last_tri["d"] == first_esa["d"]:
                    w_final = first_esa["w"]
                    if "PIOGGIA" in last_tri["w"] or "TEMPORALE" in last_tri["w"] or "NEVE" in last_tri["w"]:
                         w_final = last_tri["w"]
                    elif "PIOGGIA" in first_esa["w"] or "TEMPORALE" in first_esa["w"] or "NEVE" in first_esa["w"]:
                         w_final = first_esa["w"]
                    
                    merged_day = {
                        "d": last_tri["d"],
                        "tmin": min(last_tri["tmin"], first_esa["tmin"]),
                        "tmax": max(last_tri["tmax"], first_esa["tmax"]),
                        "p": round(last_tri["p"] + first_esa["p"], 1),
                        "w": w_final
                    }
                    final_daily[-1] = merged_day
                    final_daily.extend(daily_esa[1:])
                else:
                    final_daily.extend(daily_esa)
            else:
                final_daily.extend(daily_esa)

            # Costruzione payload
            city_data = {
                "r": run_info["run_str"], "c": city, "x": info['lat'], "y": info['lon'], "z": info['elev'],
                "TRIORARIO": trihourly_data, "ESAORARIO": esaorario_data, "GIORNALIERO": final_daily
            }

            # --- 2. ARIA (CAMS) - INIEZIONE ---
            if process_air and ds_cams is not None:
                try:
                    # Sel nearest CAMS
                    sel_c = ds_cams.sel(lat=info['lat'], lon=info['lon'], method="nearest")
                    df = sel_c.to_dataframe().reset_index()
                    
                    # Logica tempo robusta
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
                    df = df.tz_convert("Europe/Rome")
                    cutoff_dt = df.index[0] + pd.Timedelta(days=4)
                    df = df[df.index < cutoff_dt]
                    df = df[["pm25", "pm10", "no2", "o3"]]

                    # Aria aggregata
                    aria_h = clean_cams_data(df)
                    # Calcola l'offset dall'inizio del giorno
                    first_hour = df.index[0].hour
                    offset_str = f"{first_hour}h"

                    aria_3h = clean_cams_data(df.resample("3h", offset=offset_str).mean().dropna())
                    
                    df_d = df.resample("1D").mean()
                    counts = df.resample("1D").count()["pm25"]
                    df_d = df_d[counts >= 18].round(0).astype(int)
                    
                    aria_d = []
                    for dt, row in df_d.iterrows():
                        val, lbl = calculate_caqi(row)
                        aria_d.append({"d": dt.strftime("%Y%m%d"), "pm25": int(row["pm25"]), "pm10": int(row["pm10"]), "no2": int(row["no2"]), "o3": int(row["o3"]), "aqi_value": val, "aqi_class": lbl})

                    city_data["ARIA_ORARIO"] = aria_h
                    city_data["ARIA_TRIORARIO"] = aria_3h
                    city_data["ARIA_GIORNO"] = aria_d
                except Exception as e:
                    # Se fallisce l'aria (es. fuori griglia), non blocchiamo il meteo
                    pass

            # --- 3. SALVATAGGIO E UPLOAD UNICO ---
            safe_name = city.replace("'", " ").replace("/", "-")
            out_file = os.path.join(run_info["outdir"], f"{safe_name}_ecmwf.json")
            
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(city_data, f, separators=(",", ":"), ensure_ascii=False)
            
            if s3_client:
                upload_to_r2(s3_client, out_file, run_info["run_date"], run_info["run_hour"], os.path.basename(out_file))
            
            processed += 1
            if processed % 50 == 0: print(f"{processed}...", end=" ", flush=True)

        except Exception as e:
            print(f"\n⚠️ Errore {city}: {e}")
            continue
            
    print(f"\n✅ Completato {os.path.basename(venues_path)}: {processed} città.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    rd, rh = get_run_datetime_now_utc()
    run_str = f"{rd}{rh}"
    print(f"--- RUN: {run_str} ---")
    
    # 1. DOWNLOAD ALL
    files = download_data_unified(rd, rh)
    
    # 2. SETUP
    tf = TimezoneFinder(in_memory=True)
    s3 = get_r2_client()
    if s3: print("✅ R2 Connesso.")
    
    outdir = f"{WORKDIR}/{run_str}"
    os.makedirs(outdir, exist_ok=True)
    run_info = {"run_date": rd, "run_hour": rh, "run_str": run_str, "outdir": outdir}

    # 3. OPEN DATASETS
    print("📂 Apertura Dataset (Meteo + Aria)...")
    try:
        datasets = {
            "main_tri": xr.open_dataset(files["main_tri"]),
            "wind_tri": xr.open_dataset(files["wind_tri"]),
            "orog": xr.open_dataset(files["orog"]),
            "main_esa": xr.open_dataset(files["main_esa"]),
            "cams": xr.open_dataset(files["cams"]) if files.get("cams") else None
        }
        
        # Prep CAMS
        if datasets["cams"]:
            rn = {}
            for v in datasets["cams"].data_vars:
                if "pm2p5" in v: rn[v] = "pm25"
                elif "pm10" in v: rn[v] = "pm10"
                elif "no2" in v: rn[v] = "no2"
                elif "o3" in v: rn[v] = "o3"
            if rn: datasets["cams"] = datasets["cams"].rename(rn)
            if "latitude" in datasets["cams"].coords: datasets["cams"] = datasets["cams"].rename({"latitude": "lat", "longitude": "lon"})
            datasets["cams"] = datasets["cams"][["pm25", "pm10", "no2", "o3"]]

    except Exception as e:
        print(f"❌ Errore Dataset: {e}")
        return

    # 4. PROCESS LOOPS
    
    # FASE 1: ITALIA (Processa Aria = True)
    process_unified_venues(VENUES_ITALIA, datasets, run_info, s3, tf, process_air=True)
    
    # FASE 2: ESTERO (Processa Aria = False)
    process_unified_venues(VENUES_ESTERO, datasets, run_info, s3, tf, process_air=False)

    # Cleanup
    for d in datasets.values(): 
        if d: d.close()
    
    # Opzionale:
    # shutil.rmtree(f"{WORKDIR}/data_temp/{run_str}")
    print("\n🏁 FINITO.")

if __name__ == "__main__":
    main()
