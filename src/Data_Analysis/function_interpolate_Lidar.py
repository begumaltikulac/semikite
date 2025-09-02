import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

def process_lidar_1theodolite(lidar_file, theo_file, date_str, start_time_str, azimuth_offset=0.0):
    """
    Liest Lidar- und Theodolit-Daten ein, schneidet den Lidar-Zeitbereich auf die Theodolit-Zeiten zu
    und interpoliert die Lidar-Geschwindigkeit ('VEL') auf die Theodolit-Zeiten.

    Parameters
    ----------
    lidar_file : str
        Pfad zur Lidar-NetCDF-Datei
    theo_file : str
        Pfad zur Theodolit-ASCII-Datei
    date_str : str
        Datum als 'YYYY-MM-DD'
    start_time_str : str
        Startzeit (UTC+1 in Theodolit-Datei) als 'HH:MM:SS'
    azimuth_offset : float, optional
        Korrektur für Azimut (default 0.0)

    Returns
    -------
    df_theo : pandas.DataFrame
        Theodolit-Daten mit UTC-Zeiten
    ws_interp : xarray.DataArray
        Interpolierte Lidar-Daten ('VEL') auf Theodolit-Zeiten
    """

    # --- Theodolit einlesen ---
    start_time = pd.to_datetime(date_str + " " + start_time_str)

    with open(theo_file, "r") as f:
        lines = f.readlines()
    lines = lines[:-3]  # letzte drei Zeilen verwerfen

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

    # --- Lidar einlesen ---
    ds_lidar = xr.open_dataset(lidar_file)

    # Zuschneiden auf Theodolit-Zeitbereich
    t_start, t_end = df_theo["time_sec"].iloc[0], df_theo["time_sec"].iloc[-1]
    ds_lidar_cut = ds_lidar.sel(time=slice(t_start, t_end))

    ds_theo = df_theo.to_xarray()
    ds_theo = ds_theo.set_coords("time_sec")
    ds_theo = ds_theo.swap_dims({"index": "time_sec"})  # "index" durch die aktuelle Dimension ersetzen

    theo_interp = ds_theo.interp(time_sec=lidar_idx)

    # Interpolation
    theo_idx = df_theo["time_sec"].to_numpy()
    ws_interp = ds_lidar_cut["VEL"].interp(time=theo_idx)

    return df_theo, ws_interp
