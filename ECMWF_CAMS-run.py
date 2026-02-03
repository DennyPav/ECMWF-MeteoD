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
CDS_KEY = raw_key.split(":", 1)[1] if ":" in raw_key else raw_key
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
CAMS_AREA = [48, 6, 35, 19] # Solo Italia

# --- FILE INPUT ---
FILE_COMUNI_ITALIA = "comuni_italia_all.json"
FILE_COMUNI_ESTERO = "comuni_estero.json"

WORKDIR = os.getcwd()
VENUES_ITALIA = os.path.join(WORKDIR, FILE_COMUNI_ITALIA)
VENUES_ESTERO = os.path.join(WORKDIR, FILE_COMUNI_ESTERO)

# Parametri Meteo
LAPSE_DRY, LAPSE_MOIST, LAPSE_P, G, RD = 0.0098, 0.006, 0.012, 9.80665, 287.05

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
    if not R2_ACCESS_KEY or "INSERISCI" in R2_ACCESS_KEY: return None
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
        print(f"❌ Upload R2 fallito: {e}")
        return False

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 6: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 18: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def get_local_time(dt_utc, lat, lon, tf_instance):
    if dt_utc.tzinfo is None: dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    timezone_str = tf_instance.timezone_at(lng=lon, lat=lat)
    if timezone_str is None: return dt_utc
    try: return dt_utc.astimezone(ZoneInfo(timezone_str))
    except Exception: return dt_utc

# --- FISICA METEO ---
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

def wet_bulb_celsius(t_c, rh_percent):
    return t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) - 4.686035

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
    p_min, p_mod, p_int = (0.3, 5.0, 20.0) if timestep_hours == 3 else (0.3, 10.0, 30.0)

    if mucape > 400 and tp_rate > 0.5 * timestep_hours: return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} TEMPORALE"
    
    if tp_rate > 0.9:
        ptype = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
        intensity = "INTENSA" if tp_rate >= p_int else "MODERATA" if tp_rate >= p_mod else "DEBOLE"
        return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} {ptype} {intensity}"
    elif 0.5 <= tp_rate <= 0.9:
        ptype = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"
        return f"{'POCO NUVOLOSO' if cloud == 'SERENO' else cloud} {ptype}"
    
    if tp_rate < 0.5:
        if t2m < season_thresh["fog_max_t"] and rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"]: return "NEBBIA"
        if t2m < season_thresh["fog_max_t"] and rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"]: return "FOSCHIA"
    return cloud

def calculate_daily_summaries(records, clct_arr, tp_arr):
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
        
        octas = np.mean(clct_arr[[x[0] for x in items]]) / 100.0 * 8
        c_state = "SERENO" if octas <= 2 else "POCO NUVOLOSO" if octas <= 4 else "NUVOLOSO" if octas <= 6 else "COPERTO"
        
        w_str = c_state
        if has_storm: w_str = f"{'POCO NUVOLOSO' if c_state == 'SERENO' else c_state} TEMPORALE"
        elif has_precip:
            ptype = "NEVE" if snow_cnt > (len(recs)-snow_cnt) else "PIOGGIA"
            pint = "INTENSA" if tp_tot >= 30 else "MODERATA" if tp_tot >= 10 else "DEBOLE"
            w_str = f"{'POCO NUVOLOSO' if c_state == 'SERENO' else c_state} {ptype} {pint}"
            
        daily.append({"d": d, "tmin": round(min(temps), 1), "tmax": round(max(temps), 1), "p": round(tp_tot, 1), "w": w_str})
    return daily

# --- FISICA ARIA ---
def calculate_caqi(row):
    def get_si(val, g):
        for i in range(len(g)-1):
            if g[i][0] <= val <= g[i+1][0]: return g[i][1] + (val - g[i][0]) * (g[i+1][1] - g[i][1]) / (g[i+1][0] - g[i][0])
        return g[-1][1] + (val - g[-1][0])
    
    idx = max(
        get_si(row["pm10"], [(0,0), (25,25), (50,50), (90,75), (180,100)]),
        get_si(row["pm25"], [(0,0), (15,25), (30,50), (55,75), (110,100)]),
        get_si(row["no2"],  [(0,0), (50,25), (100,50), (200,75), (400,100)]),
        get_si(row["o3"],   [(0,0), (60,25), (120,50), (180,75), (240,100)])
    )
    val = int(round(idx))
    lbl = "Molto Basso" if val < 25 else "Basso" if val < 50 else "Medio" if val < 75 else "Alto" if val < 100 else "Molto Alto"
    return val, lbl

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
# DOWNLOAD & DATASET MGMT
# ============================================================================

def convert_grib_to_nc_global(infile):
    if os.path.exists(infile.replace(".grib", ".nc")): return infile.replace(".grib", ".nc")
    ds = xr.open_dataset(infile, engine="cfgrib")
    outfile = infile.replace(".grib", ".nc")
    ds.to_netcdf(outfile)
    ds.close()
    return outfile

def download_data(run_date, run_hour):
    print(f"--- START DOWNLOAD: {run_date}{run_hour} ---")
    base_dir = f"{WORKDIR}/data_temp/{run_date}{run_hour}"
    os.makedirs(base_dir, exist_ok=True)
    
    # --- 1. ECMWF ---
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    steps_tri = list(range(0, 145, 3))
    steps_esa = list(range(150, 331, 6)) if run_hour == "00" else list(range(150, 319, 6))
    
    files = {
        "main_tri": (f"{base_dir}/ecmwf_main_tri.grib", ["2t", "2d", "tcc", "msl", "tp", "mucape"], steps_tri),
        "wind_tri": (f"{base_dir}/ecmwf_wind_tri.grib", ["10u", "10v"], steps_tri),
        "orog": (f"{base_dir}/ecmwf_orog.grib", ["z"], [0]),
        "main_esa": (f"{base_dir}/ecmwf_main_esa.grib", ["2t", "2d", "tcc", "msl", "tp", "mucape"], steps_esa)
    }
    
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
            print("⬇️ CAMS Italia...")
            today = datetime.now(timezone.utc).date()
            c_client = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
            req = {
                "variable": ["nitrogen_dioxide", "ozone", "particulate_matter_2.5um", "particulate_matter_10um"],
                "model": ["ensemble"], "level": ["0"], "date": [f"{today}/{today}"],
                "type": ["forecast"], "time": ["00:00"], "leadtime_hour": [str(i) for i in range(97)],
                "data_format": "netcdf_zip", "area": CAMS_AREA
            }
            try:
                c_client.retrieve("cams-europe-air-quality-forecasts", req).download(cams_zip)
                with zipfile.ZipFile(cams_zip, "r") as z: z.extractall(base_dir)
                found = [f for f in os.listdir(base_dir) if f.endswith(".nc") and "cams" not in f][0]
                os.rename(os.path.join(base_dir, found), cams_nc)
            except Exception as e:
                print(f"⚠️ CAMS fallito: {e}")
                cams_nc = None
        else:
            cams_nc = None
    
    nc_files["cams"] = cams_nc
    return nc_files

# ============================================================================
# PROCESSING UNIFICATO
# ============================================================================

def process_unified(venues_path, datasets, run_info, tf, s3, process_air=False):
    if not os.path.exists(venues_path): return
    with open(venues_path, 'r', encoding='utf-8') as f: venues = json.load(f)
    print(f"\n🚀 Elaborazione Unificata: {os.path.basename(venues_path)} ({len(venues)} città)")

    ref_dt = datetime.strptime(run_info["run_date"] + run_info["run_hour"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    _, season_thresh = get_season_precise(ref_dt)
    
    # Shortcuts datasets
    d_tri, d_wnd, d_orog, d_esa = datasets["main_tri"], datasets["wind_tri"], datasets["orog"], datasets["main_esa"]
    d_cams = datasets.get("cams")

    processed = 0
    for city, v in venues.items():
        try:
            lat, lon, elev = float(v[0]), float(v[1]), float(v[2])
            loc = {"latitude": lat, "longitude": lon}
            
            # --- 1. METEO (ECMWF) ---
            sel_tri = d_tri.sel(loc, method="nearest")
            sel_wnd = d_wnd.sel(loc, method="nearest")
            sel_esa = d_esa.sel(loc, method="nearest")
            z_mod = d_orog.sel(loc, method="nearest")["z"].values/9.81
            
            # Calcoli Triorari
            t2m = kelvin_to_celsius(sel_tri["t2m"].values)
            rh = relative_humidity(sel_tri["t2m"].values, sel_tri["d2m"].values)
            t_corr, p_corr = altitude_correction(t2m, rh, z_mod, elev, sel_tri["msl"].values/100)
            u, v_wind = sel_wnd["u10"].values, sel_wnd["v10"].values
            spd, dirs = wind_speed_direction(u, v_wind)
            tp_rate = np.diff(sel_tri["tp"].values, prepend=sel_tri["tp"].values[0]) * 1000
            
            tri_data = []
            for i in range(len(t2m)):
                dt_loc = get_local_time(ref_dt + timedelta(hours=i*3), lat, lon, tf)
                w = classify_weather(t_corr[i], rh[i], sel_tri["tcc"].values[i]*100, tp_rate[i], mps_to_kmh(spd[i]), sel_tri["mucape"].values[i], season_thresh, 3)
                tri_data.append({"d": dt_loc.strftime("%Y%m%d"), "h": dt_loc.strftime("%H"), "t": round(float(t_corr[i]),1), "r": round(float(rh[i])), "p": round(float(tp_rate[i]),1), "pr": int(p_corr[i]), "v": round(float(mps_to_kmh(spd[i])),1), "vd": wind_dir_to_cardinal(dirs[i]), "w": w})
            
            # Calcoli Esaorari
            t2m_e = kelvin_to_celsius(sel_esa["t2m"].values)
            rh_e = relative_humidity(sel_esa["t2m"].values, sel_esa["d2m"].values)
            t_corr_e, p_corr_e = altitude_correction(t2m_e, rh_e, z_mod, elev, sel_esa["msl"].values/100)
            tp_rate_e = np.diff(sel_esa["tp"].values, prepend=sel_esa["tp"].values[0]) * 1000
            
            esa_data = []
            for i in range(len(t2m_e)):
                dt_loc = get_local_time(ref_dt + timedelta(hours=150 + i*6), lat, lon, tf)
                w = classify_weather(t_corr_e[i], rh_e[i], sel_esa["tcc"].values[i]*100, tp_rate_e[i], 5.0, sel_esa["mucape"].values[i], season_thresh, 6)
                esa_data.append({"d": dt_loc.strftime("%Y%m%d"), "h": dt_loc.strftime("%H"), "t": round(float(t_corr_e[i]),1), "r": round(float(rh_e[i])), "p": round(float(tp_rate_e[i]),1), "pr": int(p_corr_e[i]), "v": None, "vd": None, "w": w})

            # Merge Giornaliero
            d_tri_sum = calculate_daily_summaries(tri_data, sel_tri["tcc"].values*100, tp_rate)
            d_esa_sum = calculate_daily_summaries(esa_data, sel_esa["tcc"].values*100, tp_rate_e)
            
            final_daily = list(d_tri_sum)
            if d_tri_sum and d_esa_sum:
                if d_tri_sum[-1]["d"] == d_esa_sum[0]["d"]:
                    d1, d2 = d_tri_sum[-1], d_esa_sum[0]
                    merged = {"d": d1["d"], "tmin": min(d1["tmin"], d2["tmin"]), "tmax": max(d1["tmax"], d2["tmax"]), "p": round(d1["p"]+d2["p"], 1), "w": d1["w"] if "PIOGGIA" in d1["w"] or "TEMPORALE" in d1["w"] else d2["w"]}
                    final_daily[-1] = merged
                    final_daily.extend(d_esa_sum[1:])
                else: final_daily.extend(d_esa_sum)
            else: final_daily.extend(d_esa_sum)

            payload = {
                "r": run_info["run_str"], "c": city, "x": lat, "y": lon, "z": elev,
                "TRIORARIO": tri_data, "ESAORARIO": esa_data, "GIORNALIERO": final_daily
            }

            # --- 2. ARIA (CAMS) - SOLO SE RICHIESTO E DISPONIBILE ---
            if process_air and d_cams is not None:
                try:
                    # Sel CAMS (gestione lat/lon diverse)
                    sel_c = d_cams.sel(lat=lat, lon=lon, method="nearest")
                    df = sel_c.to_dataframe().reset_index()
                    
                    # Fix time axis
                    base_utc = pd.Timestamp.now(timezone.utc).normalize()
                    if "time" in df.columns and df["time"].nunique() > 1: df["t"] = df["time"]
                    elif "leadtime" in df.columns: df["t"] = base_utc + pd.to_timedelta(df["leadtime"])
                    else: df["t"] = [base_utc + pd.Timedelta(hours=h) for h in range(len(df))]
                    
                    df = df.set_index("t").sort_index()
                    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
                    df = df.tz_convert("Europe/Rome")
                    df = df[df.index < (df.index[0] + pd.Timedelta(days=4))] # Cutoff 4gg
                    df = df[["pm25", "pm10", "no2", "o3"]]

                    # Aria aggregata
                    aria_h = clean_cams_data(df)
                    aria_3h = clean_cams_data(df.resample("3h").mean().dropna())
                    
                    df_d = df.resample("1D").mean()
                    counts = df.resample("1D").count()["pm25"]
                    df_d = df_d[counts >= 18].round(0).astype(int)
                    
                    aria_d = []
                    for dt, row in df_d.iterrows():
                        val, lbl = calculate_caqi(row)
                        aria_d.append({"d": dt.strftime("%Y%m%d"), "pm25": int(row["pm25"]), "pm10": int(row["pm10"]), "no2": int(row["no2"]), "o3": int(row["o3"]), "aqi_value": val, "aqi_class": lbl})

                    payload["ARIA_ORARIO"] = aria_h
                    payload["ARIA_TRIORARIO"] = aria_3h
                    payload["ARIA_GIORNO"] = aria_d
                except Exception as e:
                    # Se fallisce l'aria (es. fuori griglia), salva comunque il meteo
                    pass

            # --- 3. SALVATAGGIO E UPLOAD (UNICA VOLTA) ---
            safe_name = city.replace("'", " ").replace("/", "-")
            out_file = os.path.join(run_info["outdir"], f"{safe_name}_ecmwf.json")
            
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
            
            if s3: upload_to_r2(s3, out_file, run_info["run_date"], run_info["run_hour"], os.path.basename(out_file))
            
            processed += 1
            if processed % 50 == 0: print(f"{processed}...", end=" ", flush=True)

        except Exception as e:
            print(f"\n⚠️ Errore {city}: {e}")

    print(f"\n✅ Completato {processed} città.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    rd, rh = get_run_datetime_now_utc()
    run_str = f"{rd}{rh}"
    print(f"--- RUN: {run_str} ---")
    
    # 1. DOWNLOAD ALL
    files = download_data(rd, rh)
    
    # 2. SETUP
    tf = TimezoneFinder(in_memory=True)
    s3 = get_r2_client()
    if s3: print("✅ R2 Connesso.")
    
    outdir = f"{WORKDIR}/{run_str}"
    os.makedirs(outdir, exist_ok=True)
    run_info = {"run_date": rd, "run_hour": rh, "run_str": run_str, "outdir": outdir}

    # 3. OPEN ALL DATASETS
    print("📂 Apertura Dataset (Meteo + Aria)...")
    try:
        ds = {
            "main_tri": xr.open_dataset(files["main_tri"]),
            "wind_tri": xr.open_dataset(files["wind_tri"]),
            "orog": xr.open_dataset(files["orog"]),
            "main_esa": xr.open_dataset(files["main_esa"]),
            "cams": xr.open_dataset(files["cams"]) if files.get("cams") else None
        }
        
        # Prep CAMS names if loaded
        if ds["cams"]:
            rn = {}
            for v in ds["cams"].data_vars:
                if "pm2p5" in v: rn[v] = "pm25"
                elif "pm10" in v: rn[v] = "pm10"
                elif "no2" in v: rn[v] = "no2"
                elif "o3" in v: rn[v] = "o3"
            ds["cams"] = ds["cams"].rename(rn)
            if "latitude" in ds["cams"].coords: ds["cams"] = ds["cams"].rename({"latitude": "lat", "longitude": "lon"})
            ds["cams"] = ds["cams"][["pm25", "pm10", "no2", "o3"]]

    except Exception as e:
        print(f"❌ Errore Dataset: {e}")
        return

    # 4. PROCESS LOOP
    # Italia -> Meteo + Aria
    process_unified(VENUES_ITALIA, ds, run_info, tf, s3, process_air=True)
    
    # Estero -> Solo Meteo
    process_unified(VENUES_ESTERO, ds, run_info, tf, s3, process_air=False)

    # Cleanup
    for d in ds.values(): 
        if d: d.close()
    
    # Opzionale: Pulizia temp
    # shutil.rmtree(f"{WORKDIR}/data_temp/{run_str}")
    print("\n🏁 FINITO.")

if __name__ == "__main__":
    main()
