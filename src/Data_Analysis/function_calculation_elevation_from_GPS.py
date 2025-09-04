import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta

def process_elevation_gps(
    gps_file: str,
    lidar_file: str,
    theo_file: str,
    alt_file: str,
    date_str: str,
    start_time_str: str,
    output_csv: str,
):
    """
    Calculates the elevation angle from GPS data. Uses theodolite start time,
    and lidar timesteps, and altitude data calculated from the pressure. Results are saved as CSV.

    Parameters
    ----------
    gps_file : str
        Path to the GPS Excel file.
    lidar_file : str
        Path to the lidar NetCDF file.
    theo_file : str
        Path to the theodolite text file.
    alt_file : str
        Path to the altitude CSV file (e.g. calculated from WXT).
    date_str : str
        Date (YYYY-MM-DD).
    start_time_str : str
        Start time in format "HH:MM:SS".
    output_csv : str
        Path to the output CSV file.
    """
    # --- Read theodolite data ---
    start_time = pd.to_datetime(date_str + " " + start_time_str)

    with open(theo_file, "r") as f:
        lines = f.readlines()
    lines = lines[:-3]  # discard the last three lines

    time_sec, value1, value2 = [], [], []
    for line in lines:
        line = line.strip()
        if line.startswith("D"):
            parts = line.split()
            time_sec.append(float(parts[1]))
            value1.append(float(parts[2]))
            value2.append(float(parts[3]))
        elif line.startswith("S"):
            print("Metadata:", line)

    azimuth = np.array(value1)
    azimuth[azimuth > 360] -= 360

    df_theo = pd.DataFrame({
        "time_sec": [start_time + timedelta(seconds=s) for s in time_sec],
        "azimuth": azimuth,
        "elevation": value2
    })

    # --- Read data ---
    df_gps = pd.read_excel(gps_file)
    ds_lidar = xr.open_dataset(lidar_file)
    altitude = pd.read_csv(alt_file)
    altitude["time"] = pd.to_datetime(altitude["time"])

    # --- Convert GPS time and prepare xarray ---
    df_gps['Time'] = pd.to_datetime(df_gps['Time'][1:])
    t_start, t_end = df_gps['Time'].iloc[1], df_gps['Time'].iloc[-1]

    ds_lidar_cut = ds_lidar.sel(time=slice(t_start, t_end))
    lidar_idx = ds_lidar_cut['time']

    ds_gps = df_gps.to_xarray()
    ds_gps = ds_gps.set_coords("Time")
    ds_gps = ds_gps.swap_dims({"index": "Time"})
    _, index = np.unique(ds_gps["Time"], return_index=True)
    ds_gps = ds_gps.isel(Time=index)
    ds_gps = ds_gps.drop_isel(Time=-1)
    ds_gps = ds_gps.rename({'Time': 'time'})

    ds_gps_interp = ds_gps.interp(time=lidar_idx)

    # --- Convert altitude data ---
    ds_altitude = altitude.to_xarray()
    ds_altitude = ds_altitude.set_coords("time")
    ds_altitude = ds_altitude.swap_dims({"index": "time"})

    # --- Crop to theodolite times ---
    theo_start, theo_end = df_theo["time_sec"].iloc[0], df_theo["time_sec"].iloc[-1]
    ds_gps_interp_cut = ds_gps_interp.sel(time=slice(theo_start, theo_end))
    ds_altitude_cut = ds_altitude.sel(time=slice(theo_start, theo_end))

    # --- Location of the kite attachment ---
    lat_kite = 54.528697
    lon_kite = 11.060892

    def distance_m(lat1, lon1, lat2=lat_kite, lon2=lon_kite):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        # Conversion to meters (numpy version)
        lat_m = dlat * 111320
        lon_m = dlon * 111320 * np.cos(np.deg2rad(lat1))
        return np.sqrt(lat_m**2 + lon_m**2)

    hor_dist = distance_m(
        np.float64(ds_gps_interp_cut['Latitude'].values),
        np.float64(ds_gps_interp_cut['Longitude'].values)
    )

    elevation_angle = np.arctan(ds_altitude_cut['altitude'] / hor_dist)
    elevation_angle = np.rad2deg(elevation_angle)

    # --- Save to CSV ---
    df_elv_GPS = pd.DataFrame({
        "time": ds_gps_interp_cut['time'].values,
        "elevation_angle": elevation_angle
    })
    df_elv_GPS.to_csv(output_csv, index=False)
    print(f"Saved elevation data → {output_csv}")