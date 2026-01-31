#!/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client
import boto3

# --- GESTIONE FUSI ORARI DINAMICI ---
# Utilizziamo TimezoneFinder per trovare il fuso orario esatto basato su Lat/Lon
# Utile se nel JSON ci sono città estere (es. Tokyo, New York).
from timezonefinder import TimezoneFinder
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ---------------------- CONFIGURAZIONE ----------------------
WORKDIR = os.getcwd()
# File contenente le coordinate delle città [lat, lon, elev]
VENUES_PATH = os.path.join(WORKDIR, "comuni_italia_all.json") 

# --- CONFIGURAZIONE R2 (S3 COMPATIBLE) ---
# Le credenziali vengono lette dalle variabili d'ambiente per sicurezza
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_BUCKET_NAME = "json-meteod"

# Coefficienti per la correzione della temperatura in base all'altitudine (Lapse Rates)
LAPSE_DRY = 0.0098   # Aria secca
LAPSE_MOIST = 0.006  # Aria umida
LAPSE_P = 0.012      # Per la pressione

# SOGLIE STAGIONALI
# Usate per determinare la probabilità di nebbia (fog) o foschia (haze)
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 96, "haze_rh": 85, "fog_wind": 7.0, "haze_wind": 12.0, "fog_max_t": 15.0},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 97, "haze_rh": 85, "fog_wind": 6.0, "haze_wind": 10.0, "fog_max_t": 20.0},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 98, "haze_rh": 90, "fog_wind": 4.0, "haze_wind": 9.0, "fog_max_t": 26.0},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 95, "haze_rh": 88, "fog_wind": 7.0, "haze_wind": 11.0, "fog_max_t": 20.0}
}

# ---------------------- FUNZIONI R2 / S3 ----------------------
def get_r2_client():
    """Inizializza il client S3 per Cloudflare R2."""
    if not R2_ACCESS_KEY or not R2_SECRET_KEY or not R2_ENDPOINT:
        return None
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto'
    )

def upload_to_r2(local_file_path, run_date, run_hour, comune_name):
    """
    Carica il file JSON generato su R2.
    Percorso remoto: ECMWF/YYYYMMDDRR/nome_file.json
    """
    s3 = get_r2_client()
    if not s3: return False
    
    try:
        folder_name = f"{run_date}{run_hour}" 
        filename_only = os.path.basename(local_file_path)
        object_key = f"ECMWF/{folder_name}/{filename_only}"
        
        s3.upload_file(
            local_file_path,
            R2_BUCKET_NAME,
            object_key,
            ExtraArgs={
                'ContentType': 'application/json',
                'CacheControl': 'public, max-age=3600' # Cache di 1 ora
            }
        )
        return True
    except Exception as e:
        print(f"[R2] ❌ Errore upload {comune_name}: {e}", flush=True)
        return False

# ---------------------- UTILITY METEO & TEMPO ----------------------
def get_local_time(dt_utc, lat, lon, tf_instance):
    """
    Converte un datetime UTC nel fuso orario locale della città specificata.
    Usa TimezoneFinder per determinare il fuso in base alle coordinate.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
    # Trova la stringa del fuso orario (es. "Asia/Tokyo", "Europe/Rome")
    timezone_str = tf_instance.timezone_at(lng=lon, lat=lat)
    
    if timezone_str is None:
        return dt_utc # Fallback a UTC se in mezzo all'oceano
        
    try:
        local_tz = ZoneInfo(timezone_str)
        return dt_utc.astimezone(local_tz)
    except Exception:
        return dt_utc

def wet_bulb_celsius(t_c, rh_percent):
    """Calcola la temperatura di bulbo umido (utile per distinguere neve/pioggia)."""
    tw = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) \
         + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) \
         + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) \
         - 4.686035
    return tw

def get_run_datetime_now_utc():
    """Determina quale corsa del modello (00 o 12) scaricare in base all'ora attuale."""
    now = datetime.now(timezone.utc)
    if now.hour < 6:
        # Prima delle 6 di mattina, prendiamo la corsa 12 del giorno prima
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 18:
        # Tra le 6 e le 18, prendiamo la corsa 00 di oggi
        return now.strftime("%Y%m%d"), "00"
    # Dopo le 18, prendiamo la corsa 12 di oggi
    return now.strftime("%Y%m%d"), "12"

# ---------------------- DOWNLOAD DATI ----------------------
def convert_grib_to_nc_no_crop(infile):
    """
    Converte GRIB in NetCDF senza ritagliare le coordinate.
    Necessario per supportare città mondiali (non solo Italia).
    """
    ds = xr.open_dataset(infile, engine="cfgrib")
    outfile = infile.replace(".grib", ".nc")
    ds.to_netcdf(outfile) 
    ds.close()
    return outfile

def download_ecmwf_triorario(run_date, run_hour):
    """
    Scarica i dati triorari (step 0, 3, 6 ... 144).
    """
    steps_tri = list(range(0, 145, 3)) 
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    
    main_file = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_file = f"{grib_dir}/ecmwf_wind_tri.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"
    
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    
    # Parametri: 2t (temp), 2d (dewpoint), tcc (nuvole), msl (pressione), tp (precip), mucape (energia temporali)
    if not os.path.exists(main_file) or os.path.getsize(main_file)<30_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file)
    
    # Parametri vento: 10u, 10v
    if not os.path.exists(wind_file) or os.path.getsize(wind_file)<5_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["10u","10v"], target=wind_file)
    
    # Orografia (Z) per correzione altitudine
    if not os.path.exists(orog_file) or os.path.getsize(orog_file)<1_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=[0], param=["z"], target=orog_file)

    # Conversione in NetCDF
    main_file = convert_grib_to_nc_no_crop(main_file)
    wind_file = convert_grib_to_nc_no_crop(wind_file)
    orog_file = convert_grib_to_nc_no_crop(orog_file)
    
    return main_file, wind_file, orog_file

# NOTA: La funzione download_ecmwf_esaorario è stata rimossa perché non richiesta.

# ---------------------- LOGICA FISICA ----------------------
def kelvin_to_celsius(k): return k-273.15
def mps_to_kmh(mps): return mps*3.6

def relative_humidity(t2m_k, td2m_k):
    """Calcola l'umidità relativa da Temp e Dewpoint."""
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67*t_c)/(t_c+243.5))
    e = 6.112 * np.exp((17.67*td_c)/(td_c+243.5))
    return np.clip(100*e/es, 0, 100)

def wind_speed_direction(u,v):
    """Calcola velocità e direzione (gradi) dai vettori U e V."""
    speed_ms = np.sqrt(u**2 + v**2)
    deg = (np.degrees(np.arctan2(-u,-v))%360)
    return speed_ms, deg

def wind_dir_to_cardinal(deg):
    return ['N','NE','E','SE','S','SW','W','NW'][int((deg+22.5)%360//45)]

def get_season_precise(dt_utc):
    day_of_year = dt_utc.timetuple().tm_yday
    for season, thresh in SEASON_THRESHOLDS.items():
        if thresh["start_day"]<=day_of_year<=thresh["end_day"]:
            return season, thresh
    return "winter", SEASON_THRESHOLDS["winter"]

G = 9.80665
RD = 287.05
def altitude_correction(t2m, rh, z_model, z_station, pmsl):
    """
    Corregge la temperatura e la pressione se l'altitudine della stazione 
    è diversa dall'altitudine del modello in quel punto.
    """
    delta_z = z_model-z_station
    w_moist = np.clip(rh/100.0,0,1)
    lapse_t = LAPSE_DRY*(1.0-w_moist)+LAPSE_MOIST*w_moist
    t_corr = t2m + lapse_t*delta_z
    T_mean = t_corr + 273.15
    p_corr = pmsl*np.exp(-G*z_station/(RD*T_mean))
    return t_corr, p_corr

# ---------------------- CLASSIFICAZIONE ----------------------
def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    """Genera una stringa descrittiva (es. 'POCO NUVOLOSO PIOGGIA DEBOLE')."""
    octas = clct / 100.0 * 8
    if octas <= 2: cloud_state = "SERENO"
    elif octas <= 4: cloud_state = "POCO NUVOLOSO"
    elif octas <= 6: cloud_state = "NUVOLOSO"
    else: cloud_state = "COPERTO"

    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    prec_type_high = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
    prec_type_low = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"

    # Soglie precipitazione
    prec_debole_min = 0.3
    prec_moderata_min = 5.0
    prec_intensa_min = 20.0

    # Temporali
    if mucape > 400 and tp_rate > 0.5 * timestep_hours:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} TEMPORALE"
    
    # Precipitazioni Significative
    if tp_rate > 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        if tp_rate >= prec_intensa_min: prec_intensity = "INTENSA"
        elif tp_rate >= prec_moderata_min: prec_intensity = "MODERATA"
        else: prec_intensity = "DEBOLE"
        return f"{cloud_state} {prec_type_high} {prec_intensity}"

    # Precipitazioni Deboli / Pioviggine
    elif 0.5 <= tp_rate <= 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"

    # Nebbia / Foschia
    fog_rh = season_thresh.get("fog_rh", 95)
    fog_wd = season_thresh.get("fog_wind", 8)
    fog_t  = season_thresh.get("fog_max_t", 18)
    haze_rh = season_thresh.get("haze_rh", 85)
    haze_wd = season_thresh.get("haze_wind", 12)

    if tp_rate < 0.5: # Condizioni di stabilità o pioggia trascurabile
        weather_suffix = f" {prec_type_low}" if tp_rate >= 0.1 else ""
        
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd: return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        
        if weather_suffix:
            if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
            return cloud_state + weather_suffix
        
        return cloud_state
        
    return cloud_state

def calculate_daily_summaries(records, clct_arr, tp_arr, mucape_arr, season_thresh, timestep_hours):
    """Aggrega i dati triorari in un riepilogo giornaliero (Min/Max/Meteo Dominante)."""
    daily = []
    days_map = {}
    
    # Raggruppa record per data (stringa YYYYMMDD)
    for i, rec in enumerate(records):
        days_map.setdefault(rec["d"], []).append((i, rec))
        
    for d, items in days_map.items():
        idxs = [x[0] for x in items]
        recs = [x[1] for x in items]
        
        temps = [r["t"] for r in recs]
        if not temps: continue

        t_min, t_max = min(temps), max(temps)
        tp_tot = sum([r["p"] for r in recs])
        
        snow_steps = 0
        rain_steps = 0
        has_storm = False
        
        for r in recs:
            wtxt = r.get("w", "")
            if "TEMPORALE" in wtxt: has_storm = True
            if "PIOGGIA" in wtxt or "NEVE" in wtxt:
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
        elif (snow_steps + rain_steps) > 0: # Se ha piovuto/nevicato in modo significativo
             # Qui si potrebbe affinare la logica per definire se è una giornata di pioggia
             # Basandosi sulla quantità totale (tp_tot)
            if tp_tot >= 1.0: # Se c'è almeno 1mm cumulato
                ptype = "NEVE" if is_snow_day else "PIOGGIA"
                if tp_tot >= 30: pint = "INTENSA"
                elif tp_tot >= 10: pint = "MODERATA"
                else: pint = "DEBOLE"
                
                if c_state == "SERENO": c_state = "POCO NUVOLOSO"
                weather_str = f"{c_state} {ptype} {pint}"
            
        daily.append({
            "d": d, "tmin": round(t_min,1), "tmax": round(t_max,1), 
            "p": round(tp_tot,1), "w": weather_str
        })
    return daily

# ---------------------- MAIN PROCESSING ----------------------
def process_ecmwf_data():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN_DATE_TIME=f"{run_date}{run_hour}"
    RUN=f"{RUN_DATE_TIME}"
    
    print(f"--- Inizio Elaborazione ECMWF Run: {RUN} ---")
    
    # 1. Inizializza TimezoneFinder (pesante, fallo una volta sola)
    tf = TimezoneFinder(in_memory=True)
    
    # 2. Download Dati Triorari
    main_file_tri, wind_file_tri, orog_file = download_ecmwf_triorario(run_date,run_hour)
    
    # 3. Apertura Dataset Xarray
    ds_main_tri = xr.open_dataset(main_file_tri)
    ds_wind_tri = xr.open_dataset(wind_file_tri)
    ds_orog = xr.open_dataset(orog_file)
    
    # 4. Caricamento Lista Città
    if not os.path.exists(VENUES_PATH):
        print(f"ERRORE CRITICO: File {VENUES_PATH} non trovato.")
        return
        
    with open(VENUES_PATH,'r',encoding='utf-8') as f:
        venues_raw = json.load(f)
        # Formato atteso JSON: {"NomeCittà": [lat, lon, elev], ...}
        venues = {c:{"lat":float(v[0]),"lon":float(v[1]),"elev":float(v[2])} for c,v in venues_raw.items()}
    
    print(f"Caricate {len(venues)} città.")
    
    ref_dt = datetime.strptime(RUN_DATE_TIME, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    _, season_thresh = get_season_precise(ref_dt)
    
    outdir = f"{WORKDIR}/{RUN}"
    os.makedirs(outdir, exist_ok=True)
    
    processed = 0
    
    # 5. Loop sulle città
    for city, info in venues.items():
        try:
            # Trova l'indice della griglia più vicino alla città
            # Nota: .values trasforma in numpy array per velocità
            lat_idx_tri = np.abs(ds_main_tri.latitude - info['lat']).argmin()
            lon_idx_tri = np.abs(ds_main_tri.longitude - info['lon']).argmin()
            
            # Estrazione dati puntuali (Triorario)
            t2m_k = ds_main_tri["t2m"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values
            td2m_k = ds_main_tri["d2m"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values
            tcc = ds_main_tri["tcc"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values * 100
            msl = ds_main_tri["msl"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values / 100
            tp_cum = ds_main_tri["tp"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values
            mucape = ds_main_tri["mucape"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values
            
            u10 = ds_wind_tri["u10"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values
            v10 = ds_wind_tri["v10"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values
            
            z_model = ds_orog["z"].isel(latitude=lat_idx_tri, longitude=lon_idx_tri).values / 9.81
            
            # Calcoli derivati
            rh2m = relative_humidity(t2m_k, td2m_k)
            t2m_c = kelvin_to_celsius(t2m_k)
            # Correzione altitudine
            t2m_corr, pmsl_corr = altitude_correction(t2m_c, rh2m, z_model, info['elev'], msl)
            
            # Vento
            spd_ms, wd_deg = wind_speed_direction(u10, v10)
            spd_kmh = mps_to_kmh(spd_ms)
            
            # Precipitazione istantanea (da cumulato)
            tp_rate = np.diff(tp_cum, prepend=tp_cum[0]) * 1000
            
            trihourly_data = []
            
            # Costruzione array dati
            for i in range(len(t2m_corr)):
                dt_utc = ref_dt + timedelta(hours=i*3)
                
                # --- CALCOLO ORA LOCALE PRECISA ---
                dt_local = get_local_time(dt_utc, info['lat'], info['lon'], tf)
                
                weather = classify_weather(t2m_corr[i], rh2m[i], tcc[i], tp_rate[i],
                                         spd_kmh[i], mucape[i], season_thresh, timestep_hours=3)
                
                trihourly_data.append({
                    "d": dt_local.strftime("%Y%m%d"),
                    "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_corr[i]), 1),
                    "r": round(float(rh2m[i])),
                    "p": round(float(tp_rate[i]), 1),
                    "pr": round(float(pmsl_corr[i])),
                    "v": round(float(spd_kmh[i]), 1),
                    "vd": wind_dir_to_cardinal(wd_deg[i]),
                    "w": weather
                })
            
            # Calcolo riepilogo giornaliero dai dati triorari
            daily_summaries = calculate_daily_summaries(trihourly_data, tcc, tp_rate, mucape, season_thresh, timestep_hours=3)
            
            # Struttura JSON Finale
            city_data = {
                "r": RUN,
                "c": city,
                "x": info['lat'],
                "y": info['lon'],
                "z": info['elev'],
                # "tz": ... # Si potrebbe salvare il fuso, ma non è richiesto dallo schema
                "TRIORARIO": trihourly_data,
                "GIORNALIERO": daily_summaries
                # ESAORARIO rimosso completamente
            }

            # Salvataggio su disco
            safe_city = city.replace("'", " ").replace("/", "-")
            local_path = f"{outdir}/{safe_city}_ecmwf.json"
            
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(city_data, f, separators=(",", ":"), ensure_ascii=False)
            
            # Upload su R2
            upload_to_r2(local_path, run_date, run_hour, safe_city)

            processed += 1
            if processed % 50 == 0:
                print(f"{processed}/{len(venues)} città elaborate...")

        except Exception as e:
            print(f"Errore elaborazione {city}: {e}")
            continue

    ds_main_tri.close()
    ds_wind_tri.close()
    ds_orog.close()

    print(f"Completato {RUN}: {processed}/{len(venues)} città. File salvati in {outdir}/")
    return outdir

if __name__ == "__main__":
    process_ecmwf_data()
