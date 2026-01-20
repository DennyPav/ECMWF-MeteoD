#!/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client

# ---------------------- CONFIGURAZIONE ----------------------
WORKDIR = os.getcwd()
VENUES_PATH = f"{WORKDIR}/comuni_italia_all.json"

# Lapse rates
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.012

# SOGLIE STAGIONALI
SEASON_THRESHOLDS = {
    "winter": {
        "start_day": 1, "end_day": 80, 
        "fog_rh": 96, "haze_rh": 85, 
        "fog_wind": 7.0, "haze_wind": 12.0,
        "fog_max_t": 15.0
    },
    "spring": {
        "start_day": 81, "end_day": 172, 
        "fog_rh": 97, "haze_rh": 85, 
        "fog_wind": 6.0, "haze_wind": 10.0,
        "fog_max_t": 20.0
    },
    "summer": {
        "start_day": 173, "end_day": 263, 
        "fog_rh": 98, "haze_rh": 90, 
        "fog_wind": 4.0, "haze_wind": 9.0,
        "fog_max_t": 26.0
    },
    "autumn": {
        "start_day": 264, "end_day": 365, 
        "fog_rh": 95, "haze_rh": 88, 
        "fog_wind": 7.0, "haze_wind": 11.0,
        "fog_max_t": 20.0
    }
}

CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

# ---------------------- FUNZIONI UTILI ----------------------
def utc_to_local(dt_utc):
    m, d = dt_utc.month, dt_utc.day
    if (m > 3 and m < 10) or (m == 3 and d >= 25) or (m == 10 and d <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def wet_bulb_celsius(t_c, rh_percent):
    tw = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) \
         + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) \
         + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) \
         - 4.686035
    return tw

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 6:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "00"
    elif now.hour < 18:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# ---------------------- FUNZIONI PER DOWNLOAD E RITAGLIO ----------------------
def crop_grib_italy_xarray(infile):
    """
    Ritaglia un file GRIB globale sulle coordinate dell'Italia
    e lo salva come NetCDF (più leggero e stabile).
    """
    ds = xr.open_dataset(infile, engine="cfgrib")
    ds_it = ds.sel(longitude=slice(6,19), latitude=slice(48,35))
    outfile = infile.replace(".grib", ".nc")
    ds_it.to_netcdf(outfile)
    ds.close()
    ds_it.close()
    return outfile

def download_ecmwf_triorario(run_date, run_hour):
    steps_tri = list(range(0, 144, 3))
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    main_file = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_file = f"{grib_dir}/ecmwf_wind_tri.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    if not os.path.exists(main_file) or os.path.getsize(main_file)<30_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file)
    if not os.path.exists(wind_file) or os.path.getsize(wind_file)<5_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["10u","10v"], target=wind_file)
    if not os.path.exists(orog_file) or os.path.getsize(orog_file)<1_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=[0], param=["z"], target=orog_file)

    # --- RITAGLIO XARRAY ---
    main_file = crop_grib_italy_xarray(main_file)
    wind_file = crop_grib_italy_xarray(wind_file)
    orog_file = crop_grib_italy_xarray(orog_file)

    return main_file, wind_file, orog_file

def download_ecmwf_esaorario(run_date, run_hour):
    steps_esa = list(range(144, 331, 6)) if run_hour=="00" else list(range(144, 319, 6))
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    main_file_esa = f"{grib_dir}/ecmwf_main_esa.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    if not os.path.exists(main_file_esa) or os.path.getsize(main_file_esa)<30_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_esa, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file_esa)
    if not os.path.exists(orog_file) or os.path.getsize(orog_file)<1_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=[0], param=["z"], target=orog_file)

    # --- RITAGLIO XARRAY ---
    main_file_esa = crop_grib_italy_xarray(main_file_esa)
    orog_file = crop_grib_italy_xarray(orog_file)

    return main_file_esa, orog_file

# ---------------------- CONVERSIONI ----------------------
def kelvin_to_celsius(k): return k-273.15
def mps_to_kmh(mps): return mps*3.6

def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67*t_c)/(t_c+243.5))
    e = 6.112 * np.exp((17.67*td_c)/(td_c+243.5))
    return np.clip(100*e/es, 0, 100)

def wind_speed_direction(u,v):
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

# ---------------------- CORREZIONE ALTITUDINE ----------------------
G = 9.80665
RD = 287.05
def altitude_correction(t2m, rh, z_model, z_station, pmsl):
    delta_z = z_model-z_station
    w_moist = np.clip(rh/100.0,0,1)
    lapse_t = LAPSE_DRY*(1.0-w_moist)+LAPSE_MOIST*w_moist
    t_corr = t2m + lapse_t*delta_z
    T_mean = t_corr + 273.15
    p_corr = pmsl*np.exp(-G*z_station/(RD*T_mean))
    return t_corr, p_corr

# ---------------------- CLASSIFICAZIONE METEO ----------------------
def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    
    # Pre-calcoli comuni
    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    prec_type_high = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
    prec_type_low = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"
    
    octas = clct / 100.0 * 8
    if octas <= 2: cloud_state = "SERENO"
    elif octas <= 4: cloud_state = "POCO NUVOLOSO"
    elif octas <= 6: cloud_state = "NUVOLOSO"
    else: cloud_state = "COPERTO"

    # Soglie intensità basate su timestep
    if timestep_hours == 3:
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 5.0, 20.0
    else: # 6 hours
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 10.0, 30.0

    # 1. TEMPORALE
    if mucape > 400 and tp_rate > 0.5 * timestep_hours:
        return "TEMPORALE"
    
    # 2. PRECIPITAZIONE ALTA (> 0.9 mm)
    if tp_rate > 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        
        if tp_rate >= prec_intensa_min: prec_intensity = "INTENSA"
        elif tp_rate >= prec_moderata_min: prec_intensity = "MODERATA"
        else: prec_intensity = "DEBOLE"
        
        return f"{cloud_state} {prec_type_high} {prec_intensity}"

    # 3. PRECIPITAZIONE MEDIO-BASSA (0.5 - 0.9 mm) -> No Nebbia
    elif 0.5 <= tp_rate <= 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"

    # Recupero soglie dinamiche
    fog_rh = season_thresh.get("fog_rh", 95)
    fog_wd = season_thresh.get("fog_wind", 8)
    fog_t  = season_thresh.get("fog_max_t", 18) # Default 18 se manca la chiave
    
    haze_rh = season_thresh.get("haze_rh", 85)
    haze_wd = season_thresh.get("haze_wind", 12)

    # 4. PRECIPITAZIONE BASSISSIMA (0.1 - 0.5 mm) -> Priorità Nebbia
    if 0.1 <= tp_rate < 0.5:
        # Priorità NEBBIA/FOSCHIA (con T < soglia stagionale)
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd: return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        
        # Se no nebbia -> Precipitazione debolissima
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        return f"{cloud_state} {prec_type_low}"

    # 5. NESSUNA PRECIPITAZIONE (< 0.1 mm)
    elif tp_rate < 0.1: # else implicito
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd: return "NEBBIA"
        if t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd: return "FOSCHIA"
        
        return cloud_state
        
    return cloud_state

# ---------------------- CARICAMENTO COMUNI ----------------------
def load_venues(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    venues={c:{"lat":float(v[0]),"lon":float(v[1]),"elev":float(v[2])} for c,v in data.items()}
    print(f"Caricate {len(venues)} città")
    return venues

# ---------------------- RIEPILOGO GIORNALIERO ----------------------
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
        has_significant_snow_or_rain = False
        
        for r in recs:
            wtxt = r.get("w", "")
            # Ignora pioggerella/nevischio: considera solo PIOGGIA/NEVE
            if "PIOGGIA" in wtxt or "NEVE" in wtxt:
                has_significant_snow_or_rain = True
                wb = wet_bulb_celsius(r["t"], r["r"])
                if wb < 0.5:
                    snow_steps += 1
                else:
                    rain_steps += 1
        
        is_snow_day = snow_steps > rain_steps
        
        clct_mean = np.mean(clct_arr[idxs])
        octas = clct_mean / 100.0 * 8
        if octas <= 2: c_state = "SERENO"
        elif octas <= 4: c_state = "POCO NUVOLOSO"
        elif octas <= 6: c_state = "NUVOLOSO"
        else: c_state = "COPERTO"
        
        weather_str = c_state
        
        # Aggiungi PIOGGIA/NEVE solo se c'è stata almeno un passo con stato significativo
        if has_significant_snow_or_rain:
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


# ---------------------- PROCESSAMENTO ----------------------
def process_ecmwf_data():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN_DATE_TIME=f"{run_date}{run_hour}"
    RUN=f"{RUN_DATE_TIME}"
    
    print(f"Elaborazione ECMWF {RUN}")
    
    # DOWNLOAD + RITAGLIO
    main_file_tri, wind_file_tri, orog_file = download_ecmwf_triorario(run_date,run_hour)
    main_file_esa, _ = download_ecmwf_esaorario(run_date,run_hour)
    
    # CARICAMENTO DATASET NETCDF
    ds_main_tri=xr.open_dataset(main_file_tri)
    ds_wind_tri=xr.open_dataset(wind_file_tri)
    ds_main_esa=xr.open_dataset(main_file_esa)
    ds_orog=xr.open_dataset(orog_file)
    
    venues=load_venues(VENUES_PATH)
    ref_dt=datetime.strptime(RUN_DATE_TIME,"%Y%m%d%H").replace(tzinfo=timezone.utc)
    season, season_thresh=get_season_precise(ref_dt)
    
    outdir=f"{WORKDIR}/{RUN}"
    os.makedirs(outdir,exist_ok=True)
    
    processed=0
    for city,info in venues.items():
        try:
            lat_idx_tri=np.abs(ds_main_tri.latitude-info['lat']).argmin()
            lon_idx_tri=np.abs(ds_main_tri.longitude-info['lon']).argmin()
            lat_idx_esa=np.abs(ds_main_esa.latitude-info['lat']).argmin()
            lon_idx_esa=np.abs(ds_main_esa.longitude-info['lon']).argmin()
            
            # ---------------------- TRIORARIO ----------------------
            t2m_k=ds_main_tri["t2m"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values
            td2m_k=ds_main_tri["d2m"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values
            tcc=ds_main_tri["tcc"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values*100
            msl=ds_main_tri["msl"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values/100
            tp_cum=ds_main_tri["tp"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values
            mucape=ds_main_tri["mucape"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values
            u10=ds_wind_tri["u10"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values
            v10=ds_wind_tri["v10"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values
            z_model=ds_orog["z"].isel(latitude=lat_idx_tri,longitude=lon_idx_tri).values/9.81
            
            rh2m=relative_humidity(t2m_k,td2m_k)
            t2m_c=kelvin_to_celsius(t2m_k)
            t2m_corr,pmsl_corr=altitude_correction(t2m_c,rh2m,z_model,info['elev'],msl)
            spd_ms,wd_deg=wind_speed_direction(u10,v10)
            spd_kmh=mps_to_kmh(spd_ms)
            tp_rate=np.diff(tp_cum,prepend=tp_cum[0])*1000
            
            trihourly_data=[]
            for i in range(len(t2m_corr)):
                dt_utc=ref_dt+timedelta(hours=i*3)
                dt_local=utc_to_local(dt_utc)
                weather=classify_weather(t2m_corr[i],rh2m[i],tcc[i],tp_rate[i],
                                         spd_kmh[i],mucape[i],season_thresh,timestep_hours=3)
                trihourly_data.append({
                    "d":dt_local.strftime("%Y%m%d"),
                    "h":dt_local.strftime("%H"),
                    "t":round(float(t2m_corr[i]),1),
                    "r":round(float(rh2m[i])),
                    "p":round(float(tp_rate[i]),1),
                    "pr":round(float(pmsl_corr[i])),
                    "v":round(float(spd_kmh[i]),1),
                    "vd":wind_dir_to_cardinal(wd_deg[i]),
                    "w":weather
                })
            
            daily_summaries_tri=calculate_daily_summaries(trihourly_data,tcc,tp_rate,mucape,season_thresh,timestep_hours=3)
            
            # ---------------------- ESAORARIO ----------------------
            t2m_k_esa=ds_main_esa["t2m"].isel(latitude=lat_idx_esa,longitude=lon_idx_esa).values
            td2m_k_esa=ds_main_esa["d2m"].isel(latitude=lat_idx_esa,longitude=lon_idx_esa).values
            tcc_esa=ds_main_esa["tcc"].isel(latitude=lat_idx_esa,longitude=lon_idx_esa).values*100
            msl_esa = ds_main_esa["msl"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values / 100
            tp_cum_esa = ds_main_esa["tp"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values
            mucape_esa = ds_main_esa["mucape"].isel(latitude=lat_idx_esa, longitude=lon_idx_esa).values

            rh2m_esa = relative_humidity(t2m_k_esa, td2m_k_esa)
            t2m_c_esa = kelvin_to_celsius(t2m_k_esa)
            t2m_corr_esa, pmsl_corr_esa = altitude_correction(t2m_c_esa, rh2m_esa, z_model, info['elev'], msl_esa)
            tp_rate_esa = np.diff(tp_cum_esa, prepend=tp_cum_esa[0]) * 1000

            esaorario_data = []
            for i in range(len(t2m_corr_esa)):
                dt_utc = ref_dt + timedelta(hours=144 + i*6)
                dt_local = utc_to_local(dt_utc)
                # NOTA: Per l'esaorario non scarichiamo il vento, passiamo 5.0 km/h fittizio.
                # Questo rende la nebbia dipendente solo da RH e T.
                weather = classify_weather(t2m_corr_esa[i], rh2m_esa[i], tcc_esa[i], tp_rate_esa[i],
                                           5.0, mucape_esa[i], season_thresh, timestep_hours=6)
                esaorario_data.append({
                    "d": dt_local.strftime("%Y%m%d"),
                    "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_corr_esa[i]),1),
                    "r": round(float(rh2m_esa[i])),
                    "p": round(float(tp_rate_esa[i]),1),
                    "pr": round(float(pmsl_corr_esa[i])),
                    "w": weather
                })

            daily_summaries_esa = calculate_daily_summaries(esaorario_data, tcc_esa, tp_rate_esa,
                                                            mucape_esa, season_thresh, timestep_hours=6)

            # ---------------------- UNIONE DATI ----------------------
            trihourly_all = trihourly_data + esaorario_data
            daily_all = daily_summaries_tri + daily_summaries_esa

            city_data = {
                "r": RUN,
                "c": city,
                "x": info['lat'],
                "y": info['lon'],
                "z": info['elev'],
                "TRIORARIO": trihourly_data,
                "ESAORARIO": esaorario_data,
                "GIORNALIERO": daily_all
            }

            safe_city = city.replace("'", " ")
            with open(f"{outdir}/{safe_city}_ecmwf.json", "w", encoding="utf-8") as f:
                json.dump(city_data, f, separators=(",", ":"), ensure_ascii=False)

            processed += 1
            if processed % 50 == 0:
                print(f"{processed}/{len(venues)} città elaborate")

        except Exception as e:
            print(f"{city}: {e}")
            continue

    # ---------------------- CHIUSURA DATASET ----------------------
    ds_main_tri.close()
    ds_wind_tri.close()
    ds_main_esa.close()
    ds_orog.close()

    print(f"Completato {RUN}: {processed}/{len(venues)} città → {outdir}/")
    return outdir

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    process_ecmwf_data()
