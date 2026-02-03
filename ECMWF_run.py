#!/bin/env python3

import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client
import boto3
from timezonefinder import TimezoneFinder
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ============================================================================
# AREA CONFIGURAZIONE UTENTE (Modifica qui per uso locale)
# ============================================================================

# Inserisci qui le tue credenziali Cloudflare R2 o S3 compatibili
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_BUCKET_NAME = "json-meteod"

# Nomi dei file JSON con le località (devono essere nella stessa cartella)
FILE_COMUNI_ITALIA = "comuni_italia_all.json"
FILE_COMUNI_ESTERO = "comuni_estero.json"

# ============================================================================
# FINE CONFIGURAZIONE UTENTE
# ============================================================================

WORKDIR = os.getcwd()
VENUES_ITALIA = os.path.join(WORKDIR, FILE_COMUNI_ITALIA)
VENUES_ESTERO = os.path.join(WORKDIR, FILE_COMUNI_ESTERO)

# Lapse Rates
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.012

# SOGLIE STAGIONALI
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 96, "haze_rh": 85, "fog_wind": 7.0, "haze_wind": 12.0, "fog_max_t": 15.0},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 97, "haze_rh": 85, "fog_wind": 6.0, "haze_wind": 10.0, "fog_max_t": 20.0},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 98, "haze_rh": 90, "fog_wind": 4.0, "haze_wind": 9.0, "fog_max_t": 26.0},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 95, "haze_rh": 88, "fog_wind": 7.0, "haze_wind": 11.0, "fog_max_t": 20.0}
}

# ---------------------- FUNZIONI R2 / S3 ----------------------
def get_r2_client():
    # Verifica semplice per evitare errori se le chiavi non sono impostate
    if "INSERISCI_QUI" in R2_ACCESS_KEY or not R2_ACCESS_KEY:
        print("ATTENZIONE: Credenziali R2 non impostate. I file non verranno caricati online.")
        return None
        
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto'
    )

def upload_to_r2(s3_client, local_file_path, run_date, run_hour, comune_name):
    if not s3_client: return False
    try:
        folder_name = f"{run_date}{run_hour}"
        filename_only = os.path.basename(local_file_path)
        object_key = f"ECMWF/{folder_name}/{filename_only}"
        s3_client.upload_file(
            local_file_path,
            R2_BUCKET_NAME,
            object_key,
            ExtraArgs={'ContentType': 'application/json', 'CacheControl': 'public, max-age=3600'}
        )
        return True
    except Exception as e:
        print(f"[R2] ❌ Errore upload {comune_name}: {e}", flush=True)
        return False

# ---------------------- UTILITY METEO & TEMPO ----------------------
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

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 6:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 18:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# ---------------------- DOWNLOAD DATI ----------------------
def convert_grib_to_nc_global(infile):
    """
    Converte un file GRIB globale in NetCDF senza ritagliare.
    """
    print(f"Conversione GRIB -> NC: {infile} ...")
    ds = xr.open_dataset(infile, engine="cfgrib")
    outfile = infile.replace(".grib", ".nc")
    # Usa compressione per risparmiare spazio, opzionale ma utile
    ds.to_netcdf(outfile) 
    ds.close()
    return outfile

def download_ecmwf_data(run_date, run_hour):
    print(f"--- Download Dati ECMWF per run {run_date}{run_hour} ---")
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    
    # 1. Triorario (0-144)
    steps_tri = list(range(0, 145, 3))
    main_file_tri = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_file_tri = f"{grib_dir}/ecmwf_wind_tri.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"

    if not os.path.exists(main_file_tri) or os.path.getsize(main_file_tri) < 30_000_000:
        print("Scaricamento MAIN Triorario...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["2t", "2d", "tcc", "msl", "tp", "mucape"], target=main_file_tri)

    if not os.path.exists(wind_file_tri) or os.path.getsize(wind_file_tri) < 5_000_000:
        print("Scaricamento WIND Triorario...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["10u", "10v"], target=wind_file_tri)
    
    if not os.path.exists(orog_file) or os.path.getsize(orog_file) < 1_000:
        print("Scaricamento Orografia...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=[0], param=["z"], target=orog_file)

    # 2. Esaorario (150-3xx)
    steps_esa = list(range(150, 331, 6)) if run_hour == "00" else list(range(150, 319, 6))
    main_file_esa = f"{grib_dir}/ecmwf_main_esa.grib"
    
    if not os.path.exists(main_file_esa) or os.path.getsize(main_file_esa) < 30_000_000:
        print("Scaricamento MAIN Esaorario...")
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_esa, param=["2t", "2d", "tcc", "msl", "tp", "mucape"], target=main_file_esa)

    # Conversione in NetCDF (Globali)
    # Verifichiamo se esistono già i NC per non rifare la conversione inutile
    nc_main_tri = main_file_tri.replace(".grib", ".nc")
    if not os.path.exists(nc_main_tri): nc_main_tri = convert_grib_to_nc_global(main_file_tri)
    
    nc_wind_tri = wind_file_tri.replace(".grib", ".nc")
    if not os.path.exists(nc_wind_tri): nc_wind_tri = convert_grib_to_nc_global(wind_file_tri)
    
    nc_orog = orog_file.replace(".grib", ".nc")
    if not os.path.exists(nc_orog): nc_orog = convert_grib_to_nc_global(orog_file)
    
    nc_main_esa = main_file_esa.replace(".grib", ".nc")
    if not os.path.exists(nc_main_esa): nc_main_esa = convert_grib_to_nc_global(main_file_esa)
    
    return nc_main_tri, nc_wind_tri, nc_orog, nc_main_esa

# ---------------------- CALCOLI FISICI ----------------------
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

G = 9.80665
RD = 287.05

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

# ---------------------- PROCESSO CORE (UNICO) ----------------------
def process_venues(venues_path, datasets, run_info, s3_client, tf_instance):
    """
    Processa una lista di città (venues_path) usando i dataset aperti.
    Gestisce correttamente il merge nel giorno di transizione tra triorario ed esaorario.
    """
    if not os.path.exists(venues_path):
        print(f"⚠️  File non trovato: {venues_path}. Salto questa lista.")
        return

    with open(venues_path, 'r', encoding='utf-8') as f:
        venues_raw = json.load(f)
    
    # Normalizza coordinate
    venues = {c: {"lat": float(v[0]), "lon": float(v[1]), "elev": float(v[2])} for c, v in venues_raw.items()}
    print(f"Elaborazione di {len(venues)} città da {os.path.basename(venues_path)}...")

    ref_dt = datetime.strptime(run_info["run_date"] + run_info["run_hour"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    _, season_thresh = get_season_precise(ref_dt)
    
    processed = 0
    ds_main_tri = datasets["main_tri"]
    ds_wind_tri = datasets["wind_tri"]
    ds_orog = datasets["orog"]
    ds_main_esa = datasets["main_esa"]

    for city, info in venues.items():
        try:
            # Indici Griglia (uguali per tri/wind/orog se stessa griglia)
            lat_idx = np.abs(ds_main_tri.latitude - info['lat']).argmin()
            lon_idx = np.abs(ds_main_tri.longitude - info['lon']).argmin()
            # Indici ESA
            lat_idx_esa = np.abs(ds_main_esa.latitude - info['lat']).argmin()
            lon_idx_esa = np.abs(ds_main_esa.longitude - info['lon']).argmin()

            # --- TRIORARIO ---
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

            # --- ESAORARIO ---
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

            # --- MERGE GIORNALIERO (LOGICA AGGIUNTA) ---
            final_daily = list(daily_tri)  # Copia superficiale della lista trioraria
            
            if daily_tri and daily_esa:
                last_tri = daily_tri[-1]
                first_esa = daily_esa[0]
                
                # Se il giorno finale del triorario coincide con il primo dell'esaorario
                if last_tri["d"] == first_esa["d"]:
                    # Determina il meteo peggiore (priorità a pioggia/temporale)
                    w_final = first_esa["w"]
                    if "PIOGGIA" in last_tri["w"] or "TEMPORALE" in last_tri["w"] or "NEVE" in last_tri["w"]:
                         # Se la mattina era peggiore della sera, teniamo la mattina (o logica custom)
                         # Spesso è meglio unire le stringhe o dare priorità ai fenomeni
                         w_final = last_tri["w"]
                    elif "PIOGGIA" in first_esa["w"] or "TEMPORALE" in first_esa["w"] or "NEVE" in first_esa["w"]:
                         w_final = first_esa["w"]
                    
                    merged_day = {
                        "d": last_tri["d"],
                        "tmin": min(last_tri["tmin"], first_esa["tmin"]),
                        "tmax": max(last_tri["tmax"], first_esa["tmax"]),
                        "p": round(last_tri["p"] + first_esa["p"], 1), # Somma accumuli
                        "w": w_final
                    }
                    
                    # Sostituisci l'ultimo del triorario con il merged
                    final_daily[-1] = merged_day
                    # Aggiungi il resto dell'esaorario (dal secondo elemento in poi)
                    final_daily.extend(daily_esa[1:])
                else:
                    # Nessuna sovrapposizione (raro ma possibile ai bordi), accoda tutto
                    final_daily.extend(daily_esa)
            else:
                # Fallback se una delle due liste è vuota
                final_daily.extend(daily_esa)

            # --- OUTPUT ---
            city_data = {
                "r": run_info["run_str"], "c": city, "x": info['lat'], "y": info['lon'], "z": info['elev'],
                "TRIORARIO": trihourly_data, "ESAORARIO": esaorario_data, "GIORNALIERO": final_daily
            }

            safe_city = city.replace("'", " ").replace("/", "-")
            local_path = f"{run_info['outdir']}/{safe_city}_ecmwf.json"
            
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(city_data, f, separators=(",", ":"), ensure_ascii=False)
            
            # Upload su R2 solo se il client è attivo
            if s3_client:
                upload_to_r2(s3_client, local_path, run_info["run_date"], run_info["run_hour"], safe_city)
            
            processed += 1
            if processed % 50 == 0: print(f"{processed}...", end=" ", flush=True)

        except Exception as e:
            print(f"\nErrore {city}: {e}")
            continue
    print(f"\nCompletato {os.path.basename(venues_path)}: {processed}/{len(venues)} città.")

# ---------------------- MAIN ----------------------
def main():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN_DATE_TIME = f"{run_date}{run_hour}"
    print(f"--- START MERGED RUN: {RUN_DATE_TIME} ---")

    # 1. Download Unico (Global, no crop)
    # Scarica entrambi i set di dati (triorario ed esaorario)
    nc_main, nc_wind, nc_orog, nc_esa = download_ecmwf_data(run_date, run_hour)

    # 2. Setup Condiviso
    print("Inizializzazione TimezoneFinder (richiede qualche secondo)...")
    tf = TimezoneFinder(in_memory=True)
    
    s3 = get_r2_client()
    if s3: print("Client R2 Connesso.")
    
    outdir = f"{WORKDIR}/{RUN_DATE_TIME}"
    os.makedirs(outdir, exist_ok=True)
    
    run_info = {"run_date": run_date, "run_hour": run_hour, "run_str": RUN_DATE_TIME, "outdir": outdir}

    # 3. Apertura Dataset una volta sola
    print("Apertura dataset Xarray (NetCDF)...")
    try:
        ds_main_tri = xr.open_dataset(nc_main)
        ds_wind_tri = xr.open_dataset(nc_wind)
        ds_orog = xr.open_dataset(nc_orog)
        ds_main_esa = xr.open_dataset(nc_esa)
    except Exception as e:
        print(f"ERRORE APERTURA FILE NETCDF: {e}")
        return

    datasets = {
        "main_tri": ds_main_tri,
        "wind_tri": ds_wind_tri,
        "orog": ds_orog,
        "main_esa": ds_main_esa
    }

    # 4. Esecuzione Sequenziale
    print("\n--- FASE 1: ITALIA ---")
    process_venues(VENUES_ITALIA, datasets, run_info, s3, tf)

    print("\n--- FASE 2: ESTERO ---")
    process_venues(VENUES_ESTERO, datasets, run_info, s3, tf)

    # 5. Chiusura e Pulizia
    ds_main_tri.close()
    ds_wind_tri.close()
    ds_orog.close()
    ds_main_esa.close()
    
    # ---------------------- PULIZIA FINALE ----------------------
    # print("\n--- PULIZIA FILE TEMPORANEI ---")
    # try:
    #     import shutil
        
    #     # 1. Cancella la cartella dei dati grezzi (GRIB/NetCDF) - CONSIGLIATO
    #     grib_run_dir = f"{WORKDIR}/grib_ecmwf/{RUN_DATE_TIME}"
    #     if os.path.exists(grib_run_dir):
    #         shutil.rmtree(grib_run_dir)
    #         print(f"✅ Cancellati dati grezzi: {grib_run_dir}")

    #     # 2. Cancella la cartella dei JSON generati (Opzionale, se vuoi tenerne una copia togli questa parte)
    #     if os.path.exists(outdir):
    #         shutil.rmtree(outdir)
    #         print(f"✅ Cancellati JSON generati: {outdir}")

    # except Exception as e:
    #     print(f"⚠️ Errore durante la pulizia: {e}")

    # print("\nTutto completato.")

if __name__ == "__main__":
    main()
