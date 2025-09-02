import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

def process_lidar_1theodolite(lidar_file, theo_file, date_str, start_time_str, azimuth_offset=0.0):
    """
    Reads Lidar and theodolite data, cuts the Lidar time range to the theodolite times
    and interpolates the theodolite data to the Lidar times.
    Parameters
    ----------
    lidar_file : str
        path to Lidar-NetCDF file
    theo_file : str
        path to theodolite ASCII file
    date_str : str
        date as 'YYYY-MM-DD'
    start_time_str : str
        start time (UTC+1 in theodolite file) as 'HH:MM:SS'
    azimuth_offset : float, optional
        correction for azimuth (default 0.0)

    Returns
    -------
    theo_interp : pandas.DataFrame
        interpolated theodolite data at Lidar times
    ds_lidar_cut : xarray.DataArray
        lidar data ('VEL') with UTC times
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

    azimuth = np.array(value1) + azimuth_offset
    azimuth[azimuth > 360] -= 360

    df_theo = pd.DataFrame({
        "time_sec": [start_time + timedelta(seconds=s) for s in time_sec],
        "azimuth": azimuth,
        "elevation": value2
    })

    # --- Read lidar data ---
    ds_lidar = xr.open_dataset(lidar_file)

    # Crop to theodolite time range
    t_start, t_end = df_theo["time_sec"].iloc[0], df_theo["time_sec"].iloc[-1]
    ds_lidar_cut = ds_lidar.sel(time=slice(t_start, t_end))
    lidar_idx = ds_lidar_cut['time']

    ds_theo = df_theo.to_xarray()
    ds_theo = ds_theo.set_coords("time_sec")
    ds_theo = ds_theo.swap_dims({"index": "time_sec"})  # replace "index" with the current dimension

    # interpolation
    theo_interp = ds_theo.interp(time_sec=lidar_idx)

    return ds_lidar_cut, theo_interp