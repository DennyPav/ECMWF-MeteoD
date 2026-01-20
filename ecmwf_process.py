#!/bin/env python3
import os, json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
import psutil

WORKDIR = os.getcwd()
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"
BATCH_SIZE = 10  # process 10 cities at a time

# --- Stagioni ---
SEASON_THRESHOLDS = {
    "winter": {"start_day":1, "end_day":80, "fog_rh":97, "haze_rh":90, "fog_wind":5, "haze_wind":15},
    "spring": {"start_day":81, "end_day":172, "fog_rh":95, "haze_rh":87, "fog_wind":7, "haze_wind":20},
    "summer": {"start_day":173, "end_day":263, "fog_rh":93, "haze_rh":83, "fog_wind":10, "haze_wind":25},
    "autumn": {"start_day":264, "end_day":365, "fog_rh":96, "haze_rh":88, "fog_wind":6, "haze_wind":18}
}

CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

def print_ram_usage():
    process = psutil.Process(os.getpid())
    print(f"[MEMORY] Current RAM: {process.memory_info().rss / 1e6:.1f} MB")

# ---------------- Funzioni di base ---------------- #
def kelvin_to_celsius(k): return k-273.15
def mps_to_kmh(mps): return mps*3.6

def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67*t_c)/(t_c+243.5))
    e = 6.112 * np.exp((17.67*td_c)/(td_c+243.5))
    return np.clip(100*e/es,0,100)

def wind_speed_direction(u,v):
    speed_ms = np.sqrt(u**2 + v**2)
    deg = (np.degrees(np.arctan2(-u,-v))%360)
    return speed_ms, deg

def wind_dir_to_cardinal(deg):
    return ['N','NE','E','SE','S','SW','W','NW'][int((deg+22.5)%360//45)]

def load_venues(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    venues={c:{"lat":float(v[0]),"lon":float(v[1]),"elev":float(v[2])} for c,v in data.items()}
    print(f"Caricate {len(venues)} città")
    return venues

# ---------------- Funzioni principali di processing ---------------- #
def process_batch(batch_cities, RUN_DATE_TIME, outdir):
    grib_dir = f"{WORKDIR}/grib_ecmwf/{RUN_DATE_TIME}"
    ref_dt = datetime.strptime(RUN_DATE_TIME,"%Y%m%d%H").replace(tzinfo=timezone.utc)

    season, season_thresh = get_season_precise(ref_dt)

    # Carica dataset variabile per variabile
    def load_var(varname, grib_file):
        ds = xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": varname}})
        data = ds[varname].values
        latitudes = ds.latitude.values
        longitudes = ds.longitude.values
        ds.close()
        return data, latitudes, longitudes

    print_ram_usage()
    # --- Variabili triorario ---
    t2m_data, lats_tri, lons_tri = load_var("t2m", f"{grib_dir}/ecmwf_main_tri.grib")
    d2m_data, _, _ = load_var("d2m", f"{grib_dir}/ecmwf_main_tri.grib")
    tcc_data, _, _ = load_var("tcc", f"{grib_dir}/ecmwf_main_tri.grib")
    tp_data, _, _ = load_var("tp", f"{grib_dir}/ecmwf_main_tri.grib")
    mucape_data, _, _ = load_var("mucape", f"{grib_dir}/ecmwf_main_tri.grib")
    u10_data, _, _ = load_var("u10", f"{grib_dir}/ecmwf_wind_tri.grib")
    v10_data, _, _ = load_var("v10", f"{grib_dir}/ecmwf_wind_tri.grib")
    msl_data, _, _ = load_var("msl", f"{grib_dir}/ecmwf_main_tri.grib")
    orog_data, lats_orog, lons_orog = load_var("z", f"{grib_dir}/ecmwf_orog.grib")

    print_ram_usage()

    for city, info in batch_cities.items():
        try:
            # Indici più vicini
            lat_idx_tri = np.abs(lats_tri - info['lat']).argmin()
            lon_idx_tri = np.abs(lons_tri - info['lon']).argmin()
            lat_idx_orog = np.abs(lats_orog - info['lat']).argmin()
            lon_idx_orog = np.abs(lons_orog - info['lon']).argmin()

            t2m = t2m_data[:, lat_idx_tri, lon_idx_tri]
            d2m = d2m_data[:, lat_idx_tri, lon_idx_tri]
            tcc = tcc_data[:, lat_idx_tri, lon_idx_tri]*100
            tp_cum = tp_data[:, lat_idx_tri, lon_idx_tri]
            mucape = mucape_data[:, lat_idx_tri, lon_idx_tri]
            u10 = u10_data[:, lat_idx_tri, lon_idx_tri]
            v10 = v10_data[:, lat_idx_tri, lon_idx_tri]
            msl = msl_data[:, lat_idx_tri, lon_idx_tri]/100
            z_model = orog_data[lat_idx_orog, lon_idx_orog]/9.81

            rh2m = relative_humidity(t2m, d2m)
            t2m_c = kelvin_to_celsius(t2m)
            t2m_corr, pmsl_corr = altitude_correction(t2m_c, rh2m, z_model, info['elev'], msl)
            spd_ms, wd_deg = wind_speed_direction(u10, v10)
            spd_kmh = mps_to_kmh(spd_ms)
            tp_rate = np.diff(tp_cum, prepend=tp_cum[0])*1000

            # Triorario
            trihourly_data=[]
            for i in range(len(t2m_corr)):
                dt_utc = ref_dt + timedelta(hours=i*3)
                dt_local = utc_to_local(dt_utc)
                weather = classify_weather(t2m_corr[i], rh2m[i], tcc[i], tp_rate[i],
                                           spd_kmh[i], mucape[i], season_thresh, timestep_hours=3)
                trihourly_data.append({
                    "d": dt_local.strftime("%Y%m%d"),
                    "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_corr[i]),1),
                    "r": round(float(rh2m[i])),
                    "p": round(float(tp_rate[i]),1),
                    "pr": round(float(pmsl_corr[i])),
                    "v": round(float(spd_kmh[i]),1),
                    "vd": wind_dir_to_cardinal(wd_deg[i]),
                    "w": weather
                })

            # Salva JSON città
            city_data = {"c": city, "x": info['lat'], "y": info['lon'], "z": info['elev'], "TRIORARIO": trihourly_data}
            with open(f"{outdir}/{city}.json",'w',encoding='utf-8') as f:
                json.dump(city_data,f,separators=(',',':'),ensure_ascii=False)

            print(f"[INFO] Processata città {city}")
            print_ram_usage()

        except Exception as e:
            print(f"[ERROR] {city}: {e}")
            continue

def process_ecmwf_all():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN_DATE_TIME = f"{run_date}{run_hour}"
    outdir = f"{WORKDIR}/{RUN_DATE_TIME}"
    os.makedirs(outdir, exist_ok=True)

    venues = load_venues(VENUES_PATH)
    city_list = list(venues.items())
    total = len(city_list)
    for i in range(0, total, BATCH_SIZE):
        batch_cities = dict(city_list[i:i+BATCH_SIZE])
        print(f"\n[INFO] Processing batch {i//BATCH_SIZE+1} ({len(batch_cities)} cities)")
        process_batch(batch_cities, RUN_DATE_TIME, outdir)

if __name__=="__main__":
    process_ecmwf_all()
