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
# AREA CONFIGURAZIONE UTENTE
# ============================================================================

# --- CLOUDFLARE R2 / S3 ---
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_BUCKET_NAME = "json-meteod"

# --- CAMS (COPERNICUS) ---
# Inserisci la tua chiave qui se non è nelle variabili d'ambiente
raw_key = os.environ.get("CDS_API_KEY", "c12c8903-f197-4b19-8015-963f3646bcc6") 
if ":" in raw_key:
    CDS_KEY = raw_key.split(":", 1)[1]
else:
    CDS_KEY = raw_key
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"

# Area Download CAMS (N, W, S, E) - Copre Europa e Mediterraneo allargato
CAMS_AREA = [60, -15, 30, 40] 

# --- FILE INPUT ---
FILE_COMUNI_ITALIA = "comuni_italia_all.json"
FILE_COMUNI_ESTERO = "comuni_estero.json"

# ============================================================================
# FINE CONFIGURAZIONE UTENTE
# ============================================================================

WORKDIR = os.getcwd()
VENUES_ITALIA = os.path.join(WORKDIR, FILE_COMUNI_ITALIA)
VENUES_ESTERO = os.path.join(WORKDIR, FILE_COMUNI_ESTERO)

# Lapse Rates & Parametri Meteo
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.012
G = 9.80665
RD = 287.05

SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 96, "haze_rh": 85, "fog_wind": 7.0, "haze_wind": 12.0, "fog_max_t": 15.0},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 97, "haze_rh": 85, "fog_wind": 6.0, "haze_wind": 10.0, "fog_max_t": 20.0},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 98, "haze_rh": 90, "fog_wind": 4.0, "haze_wind": 9.0, "fog_max_t": 26.0},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 95, "haze_rh": 88, "fog_wind": 7.0, "haze_wind": 11.0, "fog_max_t": 20.0}
}

# ---------------------- FUNZIONI R2 / S3 ----------------------
def get_r2_client():
    if not R2_ACCESS_KEY or "INSERISCI" in R2_ACCESS_KEY:
        print("⚠️  Credenziali R2 non impostate/valide.")
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

# ---------------------- UTILITY METEO & TEMPO ----------------------
def get_local_time(dt_utc, lat, lon, tf_instance):
    if dt_utc.tzinfo is None: dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    timezone_str = tf_instance.timezone_at(lng=lon, lat=lat)
    if timezone_str is None: return dt_utc
    try: return dt_utc.astimezone(ZoneInfo(timezone_str))
    except Exception: return dt_utc

def wet_bulb_celsius(t_c, rh_percent):
    return t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) - 4.686035

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 6: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 18: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# ---------------------- FUNZIONI CALCOLO ARIA (CAMS) ----------------------
def calculate_caqi(row):
    """Calcola AQI standard Europeo (CAQI)"""
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
    """Formatta per JSON CAMS con split data/ora"""
    df = df.round(0).astype(int)
    out_list = []
    for dt, row in df.iterrows():
        item = {
            "d": dt.strftime("%Y%m%d"),
            "h": dt.strftime("%H"),
            "pm25": int(row["pm25"]), "pm10": int(row["pm10"]),
            "no2": int(row["no2"]), "o3": int(row["o3"])
        }
        out_list.append(item)
    return out_list

# ---------------------- DOWNLOAD DATI ECMWF ----------------------
def convert_grib_to_nc_global(infile):
    ds = xr.open_dataset(infile, engine="cfgrib")
    outfile = infile.replace(".grib", ".nc")
    ds.to_netcdf(outfile) 
    ds.close()
    return outfile

def download_ecmwf_data(run_date, run_hour):
    print(f"--- Download Dati ECMWF per run {run_date}{run_hour} ---")
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    
    # 1. Triorario
    steps_tri = list(range(0, 145, 3))
    main_file_tri = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_file_tri = f"{grib_dir}/ecmwf_wind_tri.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"

    if not os.path.exists(main_file_tri) or os.path.getsize(main_file_tri) < 30_000_000:
        print("Scaricamento MAIN Triorario...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc", step=steps_tri, param=["2t", "2d", "tcc", "msl", "tp", "mucape"], target=main_file_tri)

    if not os.path.exists(wind_file_tri) or os.path.getsize(wind_file_tri) < 5_000_000:
        print("Scaricamento WIND Triorario...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc", step=steps_tri, param=["10u", "10v"], target=wind_file_tri)
    
    if not os.path.exists(orog_file) or os.path.getsize(orog_file) < 1_000:
        print("Scaricamento Orografia...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc", step=[0], param=["z"], target=orog_file)

    # 2. Esaorario
    steps_esa = list(range(150, 331, 6)) if run_hour == "00" else list(range(150, 319, 6))
    main_file_esa = f"{grib_dir}/ecmwf_main_esa.grib"
    if not os.path.exists(main_file_esa) or os.path.getsize(main_file_esa) < 30_000_000:
        print("Scaricamento MAIN Esaorario...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc", step=steps_esa, param=["2t", "2d", "tcc", "msl", "tp", "mucape"], target=main_file_esa)

    # Conversione NC
    nc_main_tri = convert_grib_to_nc_global(main_file_tri) if not os.path.exists(main_file_tri.replace(".grib", ".nc")) else main_file_tri.replace(".grib", ".nc")
    nc_wind_tri = convert_grib_to_nc_global(wind_file_tri) if not os.path.exists(wind_file_tri.replace(".grib", ".nc")) else wind_file_tri.replace(".grib", ".nc")
    nc_orog = convert_grib_to_nc_global(orog_file) if not os.path.exists(orog_file.replace(".grib", ".nc")) else orog_file.replace(".grib", ".nc")
    nc_main_esa = convert_grib_to_nc_global(main_file_esa) if not os.path.exists(main_file_esa.replace(".grib", ".nc")) else main_file_esa.replace(".grib", ".nc")
    
    return nc_main_tri, nc_wind_tri, nc_orog, nc_main_esa

# ---------------------- ELABORAZIONE METEO (CORE) ----------------------
def kelvin_to_celsius(k): return k - 273.15
def mps_to_kmh(mps): return mps * 3.6
def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip(100 * e / es, 0, 100)
def wind_speed_direction(u, v): return np.sqrt(u**2 + v**2), (np.degrees(np.arctan2(-u, -v)) % 360)
def wind_dir_to_cardinal(deg): return ['N','NE','E','SE','S','SW','W','NW'][int((deg + 22.5) % 360 // 45)]
def get_season_precise(dt_utc):
    day = dt_utc.timetuple().tm_yday
    for s, t in SEASON_THRESHOLDS.items():
        if t["start_day"] <= day <= t["end_day"]: return s, t
    return "winter", SEASON_THRESHOLDS["winter"]

def altitude_correction(t2m, rh, z_model, z_station, pmsl):
    delta_z = z_model - z_station
    w_moist = np.clip(rh / 100.0, 0, 1)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr = t2m + lapse_t * delta_z
    p_corr = pmsl * np.exp(-G * z_station / (RD * (t_corr + 273.15)))
    return t_corr, p_corr

def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    octas = clct / 100.0 * 8
    cloud = "SERENO" if octas <= 2 else "POCO NUVOLOSO" if octas <= 4 else "NUVOLOSO" if octas <= 6 else "COPERTO"
    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    prec_type_high = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
    prec_type_low = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"
    
    p_min, p_mod, p_int = (0.3, 5.0, 20.0) if timestep_hours == 3 else (0.3, 10.0, 30.0)

    if mucape > 400 and tp_rate > 0.5 * timestep_hours:
        return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} TEMPORALE"

    if tp_rate > 0.9:
        intensity = "INTENSA" if tp_rate >= p_int else "MODERATA" if tp_rate >= p_mod else "DEBOLE"
        return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} {prec_type_high} {intensity}"
    elif 0.5 <= tp_rate <= 0.9:
        return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} {prec_type_low}"
    
    if 0.1 <= tp_rate < 0.5:
        if t2m < season_thresh["fog_max_t"] and rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"]: return "NEBBIA"
        if t2m < season_thresh["fog_max_t"] and rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"]: return "FOSCHIA"
        return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} {prec_type_low}"
    elif tp_rate < 0.1:
        if t2m < season_thresh["fog_max_t"] and rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"]: return "NEBBIA"
        if t2m < season_thresh["fog_max_t"] and rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"]: return "FOSCHIA"
    return cloud

def calculate_daily_summaries(records, clct_arr, tp_arr, mucape_arr, season_thresh, timestep_hours):
    daily = []
    days_map = {}
    for i, rec in enumerate(records): days_map.setdefault(rec["d"], []).append((i, rec))
    
    for d, items in days_map.items():
        recs = [x[1] for x in items]
        temps = [r["t"] for r in recs]
        tp_tot = sum([r["p"] for r in recs])
        
        has_storm = any("TEMPORALE" in r["w"] for r in recs)
        has_precip = any("PIOGGIA" in r["w"] or "NEVE" in r["w"] for r in recs)
        snow_cnt = sum(1 for r in recs if wet_bulb_celsius(r["t"], r["r"]) < 0.5)
        rain_cnt = len(recs) - snow_cnt
        
        octas = np.mean(clct_arr[[x[0] for x in items]]) / 100.0 * 8
        c_state = "SERENO" if octas <= 2 else "POCO NUVOLOSO" if octas <= 4 else "NUVOLOSO" if octas <= 6 else "COPERTO"
        
        w_str = c_state
        if has_storm: w_str = f"{'POCO NUVOLOSO' if c_state == 'SERENO' else c_state} TEMPORALE"
        elif has_precip:
            ptype = "NEVE" if snow_cnt > rain_cnt else "PIOGGIA"
            pint = "INTENSA" if tp_tot >= 30 else "MODERATA" if tp_tot >= 10 else "DEBOLE"
            w_str = f"{'POCO NUVOLOSO' if c_state == 'SERENO' else c_state} {ptype} {pint}"
            
        daily.append({"d": d, "tmin": round(min(temps), 1), "tmax": round(max(temps), 1), "p": round(tp_tot, 1), "w": w_str})
    return daily

def process_venues(venues_path, datasets, run_info, tf_instance):
    if not os.path.exists(venues_path): return
    with open(venues_path, 'r', encoding='utf-8') as f: venues_raw = json.load(f)
    venues = {c: {"lat": float(v[0]), "lon": float(v[1]), "elev": float(v[2])} for c, v in venues_raw.items()}
    print(f"Elaborazione Meteo {len(venues)} città da {os.path.basename(venues_path)}...")

    ref_dt = datetime.strptime(run_info["run_date"] + run_info["run_hour"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    _, season_thresh = get_season_precise(ref_dt)
    
    ds_main_tri, ds_wind_tri, ds_orog, ds_main_esa = datasets["main_tri"], datasets["wind_tri"], datasets["orog"], datasets["main_esa"]
    processed = 0

    for city, info in venues.items():
        try:
            # Selezione nearest (ottimizzata)
            loc = {"latitude": info['lat'], "longitude": info['lon']}
            d_tri = ds_main_tri.sel(loc, method="nearest")
            d_wnd = ds_wind_tri.sel(loc, method="nearest")
            d_orog = ds_orog.sel(loc, method="nearest")
            d_esa = ds_main_esa.sel(loc, method="nearest")
            
            # Dati Triorari
            t2m_c, pmsl_corr = altitude_correction(kelvin_to_celsius(d_tri["t2m"].values), relative_humidity(d_tri["t2m"].values, d_tri["d2m"].values), d_orog["z"].values/9.81, info['elev'], d_tri["msl"].values/100)
            spd_ms, wd_deg = wind_speed_direction(d_wnd["u10"].values, d_wnd["v10"].values)
            tp_rate = np.diff(d_tri["tp"].values, prepend=d_tri["tp"].values[0]) * 1000
            
            tri_data = []
            for i in range(len(t2m_c)):
                dt_loc = get_local_time(ref_dt + timedelta(hours=i*3), info['lat'], info['lon'], tf_instance)
                w = classify_weather(t2m_c[i], relative_humidity(d_tri["t2m"].values[i], d_tri["d2m"].values[i]), d_tri["tcc"].values[i]*100, tp_rate[i], mps_to_kmh(spd_ms[i]), d_tri["mucape"].values[i], season_thresh, 3)
                tri_data.append({"d": dt_loc.strftime("%Y%m%d"), "h": dt_loc.strftime("%H"), "t": round(float(t2m_c[i]), 1), "r": round(float(relative_humidity(d_tri["t2m"].values[i], d_tri["d2m"].values[i]))), "p": round(float(tp_rate[i]), 1), "pr": round(float(pmsl_corr[i])), "v": round(float(mps_to_kmh(spd_ms[i])), 1), "vd": wind_dir_to_cardinal(wd_deg[i]), "w": w})
            
            # Dati Esaorari
            t2m_c_e, pmsl_corr_e = altitude_correction(kelvin_to_celsius(d_esa["t2m"].values), relative_humidity(d_esa["t2m"].values, d_esa["d2m"].values), d_orog["z"].values/9.81, info['elev'], d_esa["msl"].values/100)
            tp_rate_e = np.diff(d_esa["tp"].values, prepend=d_esa["tp"].values[0]) * 1000
            
            esa_data = []
            for i in range(len(t2m_c_e)):
                dt_loc = get_local_time(ref_dt + timedelta(hours=150 + i*6), info['lat'], info['lon'], tf_instance)
                w = classify_weather(t2m_c_e[i], relative_humidity(d_esa["t2m"].values[i], d_esa["d2m"].values[i]), d_esa["tcc"].values[i]*100, tp_rate_e[i], 5.0, d_esa["mucape"].values[i], season_thresh, 6)
                esa_data.append({"d": dt_loc.strftime("%Y%m%d"), "h": dt_loc.strftime("%H"), "t": round(float(t2m_c_e[i]), 1), "r": round(float(relative_humidity(d_esa["t2m"].values[i], d_esa["d2m"].values[i]))), "p": round(float(tp_rate_e[i]), 1), "pr": round(float(pmsl_corr_e[i])), "v": None, "vd": None, "w": w})

            # Merge Giornaliero
            daily_tri = calculate_daily_summaries(tri_data, d_tri["tcc"].values*100, tp_rate, d_tri["mucape"].values, season_thresh, 3)
            daily_esa = calculate_daily_summaries(esa_data, d_esa["tcc"].values*100, tp_rate_e, d_esa["mucape"].values, season_thresh, 6)
            final_daily = list(daily_tri)
            if daily_tri and daily_esa:
                if daily_tri[-1]["d"] == daily_esa[0]["d"]:
                    dt, de = daily_tri[-1], daily_esa[0]
                    merged = {"d": dt["d"], "tmin": min(dt["tmin"], de["tmin"]), "tmax": max(dt["tmax"], de["tmax"]), "p": round(dt["p"]+de["p"], 1), "w": dt["w"] if any(x in dt["w"] for x in ["PIOGGIA","NEVE","TEMPORALE"]) else de["w"]}
                    final_daily[-1] = merged
                    final_daily.extend(daily_esa[1:])
                else: final_daily.extend(daily_esa)
            else: final_daily.extend(daily_esa)

            # Salva JSON preliminare (senza aria)
            city_data = {
                "r": run_info["run_str"], "c": city, "x": info['lat'], "y": info['lon'], "z": info['elev'],
                "TRIORARIO": tri_data, "ESAORARIO": esa_data, "GIORNALIERO": final_daily
            }
            safe_city = city.replace("'", " ").replace("/", "-")
            local_path = f"{run_info['outdir']}/{safe_city}_ecmwf.json"
            with open(local_path, "w", encoding="utf-8") as f: json.dump(city_data, f, separators=(",", ":"), ensure_ascii=False)
            
            processed += 1
            if processed % 50 == 0: print(f"{processed}...", end=" ", flush=True)

        except Exception as e:
            print(f"\\nErrore {city}: {e}")
            continue
    print(f"\\nCompletato Meteo {os.path.basename(venues_path)}")

# ---------------------- ELABORAZIONE ARIA (MERGE) ----------------------
def run_cams_job_and_merge(run_info, s3_client):
    """Scarica CAMS, riapre i JSON generati da ECMWF, aggiunge i dati e fa l'upload."""
    print(f"\\n--- Start CAMS Air Quality Processing ---")

    cams_dir = os.path.join(WORKDIR, "cams_temp")
    if os.path.exists(cams_dir): shutil.rmtree(cams_dir)
    os.makedirs(cams_dir, exist_ok=True)

    # 1. DOWNLOAD CAMS
    if not CDS_KEY:
        print("❌ CDS_API_KEY mancante. Salto fase CAMS.")
        return

    today_date = datetime.now(timezone.utc).date()
    # base_time_utc = pd.Timestamp(f"{today_date} 00:00:00").tz_localize("UTC")
    leadtimes = [str(i) for i in range(0, 97)]

    client = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
    request = {
        "variable": ["nitrogen_dioxide", "ozone", "particulate_matter_2.5um", "particulate_matter_10um"],
        "model": ["ensemble"], "level": ["0"], "date": [f"{today_date}/{today_date}"],
        "type": ["forecast"], "time": ["00:00"], "leadtime_hour": leadtimes,
        "data_format": "netcdf_zip", "area": CAMS_AREA,
    }

    try:
        print("⬇️  Download CAMS in corso...")
        zip_path = os.path.join(cams_dir, "cams.zip")
        client.retrieve("cams-europe-air-quality-forecasts", request).download(zip_path)
        with zipfile.ZipFile(zip_path, "r") as z: z.extractall(cams_dir)
        nc_path = [os.path.join(cams_dir, f) for f in os.listdir(cams_dir) if f.endswith(".nc")][0]
    except Exception as e:
        print(f"❌ Errore download CAMS: {e}")
        return

    # 2. OPEN NETCDF & PREPARE
    print("⚙️  Apertura NetCDF CAMS...")
    ds = xr.open_dataset(nc_path)
    rename_map = {}
    for var in ds.data_vars:
        if "pm2p5" in var.lower(): rename_map[var] = "pm25"
        elif "pm10" in var.lower(): rename_map[var] = "pm10"
        elif "no2" in var.lower(): rename_map[var] = "no2"
        elif "o3" in var.lower(): rename_map[var] = "o3"
    if rename_map: ds = ds.rename(rename_map)
    
    if "latitude" in ds.coords: ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if "level" in ds.dims: ds = ds.squeeze("level", drop=True)
    if "model" in ds.dims: ds = ds.mean("model")
    if "ensemble" in ds.dims: ds = ds.mean("ensemble")
    ds = ds[["pm25", "pm10", "no2", "o3"]]

    # Calcolo coordinate temporali
    # Nota: Usiamo time_calc per avere l'orario assoluto
    # CAMS restituisce steps/leadtimes a partire dalla data di analisi
    base_time_utc = pd.Timestamp(f"{today_date} 00:00:00").tz_localize("UTC")
    
    # Tentativo di ricostruire l'asse temporale corretto
    # Se il ds ha "time" e "step", spesso "time" è fisso e "step" varia.
    # Dobbiamo creare un indice temporale assoluto.
    if "step" in ds.coords:
        # Crea array di timestamps sommando step a base_time
        # Questo dipende da come xarray ha caricato il file (timedelta64 o altro)
        # Metodo sicuro: lavorare sul dataframe dopo sel() oppure costruire index prima
        pass 

    cutoff_date = pd.Timestamp(today_date) + pd.Timedelta(days=4)
    cutoff_dt = cutoff_date.tz_localize("Europe/Rome")

    # 3. ELABORAZIONE FILE ESISTENTI
    json_files = [f for f in os.listdir(run_info["outdir"]) if f.endswith("_ecmwf.json")]
    print(f"🔄 Injecting Air Quality into {len(json_files)} files...")

    processed_count = 0
    for jf in json_files:
        path = os.path.join(run_info["outdir"], jf)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            lat, lon = data["x"], data["y"]
            
            # Estrazione dati CAMS
            sel = ds.sel(lat=lat, lon=lon, method="nearest")
            df = sel.to_dataframe().reset_index()

            # Costruzione colonna time_calc robusta
            if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]) and df["time"].nunique() > 1:
                df["time_calc"] = df["time"]
            elif "leadtime" in df.columns:
                df["time_calc"] = base_time_utc + pd.to_timedelta(df["leadtime"])
            elif "step" in df.columns:
                 df["time_calc"] = base_time_utc + df["step"] # step è solitamente timedelta
            else:
                # Fallback se non troviamo colonne tempo esplicite
                df["time_calc"] = [base_time_utc + pd.Timedelta(hours=i) for i in range(len(df))]

            # Pulizia DataFrame
            df = df.set_index("time_calc").sort_index()
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            df = df.tz_convert("Europe/Rome")
            df = df[df.index < cutoff_dt]
            df = df[["pm25", "pm10", "no2", "o3"]]

            # --- Calcoli ---
            # 1. Orario
            aria_orario = clean_cams_data(df)

            # 2. Triorario
            df_3h = df.resample("3h").mean().dropna()
            aria_triorario = clean_cams_data(df_3h)

            # 3. Giornaliero & Previsioni
            counts = df.resample("1D").count()["pm25"]
            df_day = df.resample("1D").mean()
            valid_days = counts[counts >= 18].index # Almeno 18 ore di dati
            df_day = df_day.loc[valid_days].round(0).astype(int)

            aqi_list = []
            previsioni_list = []

            for dt, row in df_day.iterrows():
                aqi_val, aqi_label = calculate_caqi(row)
                item = {
                    "d": dt.strftime("%Y%m%d"),
                    "pm25": int(row["pm25"]), "pm10": int(row["pm10"]),
                    "no2": int(row["no2"]), "o3": int(row["o3"]),
                    "aqi_value": int(aqi_val), "aqi_class": aqi_label
                }
                aqi_list.append(item)
                
                # Struttura previsioni esplicita
                previsioni_list.append({
                    "data": item["d"], "qualita": item["aqi_class"], "valore_aqi": item["aqi_value"],
                    "dettagli": {"pm10": item["pm10"], "pm25": item["pm25"], "no2": item["no2"], "o3": item["o3"]}
                })

            # 4. Update JSON structure
            data["ARIA_ORARIO"] = aria_orario
            data["ARIA_TRIORARIO"] = aria_triorario
            data["ARIA_GIORNO"] = aqi_list
            data["PREVISIONI_ARIA"] = previsioni_list # Chiave aggiunta come richiesto precedentemente

            # 5. Save & Re-upload
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
            
            if s3_client:
                upload_to_r2(s3_client, path, run_info["run_date"], run_info["run_hour"], os.path.basename(path))

            processed_count += 1
            if processed_count % 50 == 0: print(f"{processed_count}...", end=" ", flush=True)

        except Exception as e:
            # Non blocchiamo tutto se fallisce una città
            # print(f"Err CAMS {jf}: {e}")
            pass
            
    print(f"\\n✅ CAMS Merge completato per {processed_count} files.")
    shutil.rmtree(cams_dir)


# ---------------------- MAIN ----------------------
def main():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN_DATE_TIME = f"{run_date}{run_hour}"
    print(f"--- START MERGED RUN: {RUN_DATE_TIME} ---")

    # 1. Download Unico ECMWF
    nc_main, nc_wind, nc_orog, nc_esa = download_ecmwf_data(run_date, run_hour)

    # 2. Setup Condiviso
    print("Inizializzazione TimezoneFinder...")
    tf = TimezoneFinder(in_memory=True)
    s3 = get_r2_client()
    if s3: print("Client R2 Connesso.")
    
    outdir = f"{WORKDIR}/{RUN_DATE_TIME}"
    os.makedirs(outdir, exist_ok=True)
    
    run_info = {"run_date": run_date, "run_hour": run_hour, "run_str": RUN_DATE_TIME, "outdir": outdir}

    # 3. Apertura Dataset ECMWF
    print("Apertura dataset Xarray (NetCDF)...")
    try:
        datasets = {
            "main_tri": xr.open_dataset(nc_main),
            "wind_tri": xr.open_dataset(nc_wind),
            "orog": xr.open_dataset(nc_orog),
            "main_esa": xr.open_dataset(nc_esa)
        }
    except Exception as e:
        print(f"ERRORE APERTURA FILE NETCDF: {e}")
        return

    # 4. Esecuzione Sequenziale METEO
    print("\\n--- FASE 1: METEO ITALIA ---")
    process_venues(VENUES_ITALIA, datasets, run_info, tf)

    print("\\n--- FASE 2: METEO ESTERO ---")
    process_venues(VENUES_ESTERO, datasets, run_info, tf)

    # Chiudiamo i dataset Meteo per liberare memoria prima di CAMS
    for ds in datasets.values(): ds.close()

    # 5. Esecuzione ARIA (Post-processing e Inject)
    print("\\n--- FASE 3: ARIA (CAMS) ---")
    run_cams_job_and_merge(run_info, s3)
    
    print("\\n✅ Tutto completato.")

if __name__ == "__main__":
    main()
