import pandas as pd
import numpy as np
import xarray as xr


def height_from_pressure(p2, p1=1007, rho=1.15, g=9.81):
    """
    Calculate height from pressure using hydrostatic approximation.
    """
    dp = p1 - p2
    dz = dp / (rho * g)
    return dz


def process_altitude(date_str, ptu_file, wxt_file, lidar_file, output_csv):
    """
    Process PTU and WXT data, interpolate to Lidar times, and calculate altitude.

    Parameters
    ----------
    date_str : str
        Date in format 'YYYY-MM-DD'
    ptu_file : str
        Path to PTU Excel file
    wxt_file : str
        Path to WXT CSV file
    lidar_file : str
        Path to Lidar NetCDF file
    output_csv : str
        Path to save calculated altitude CSV
    """
    # --- read data ---
    df_ptu = pd.read_excel(ptu_file)
    df_wxt = pd.read_csv(wxt_file, skiprows=6, delimiter=';')
    ds_lidar = xr.open_dataset(lidar_file)

    # --- convert PTU time and prepare xarray ---
    df_ptu['Time'] = pd.to_datetime(df_ptu['Time'][1:])
    t_start, t_end = df_ptu['Time'].iloc[1], df_ptu['Time'].iloc[-1]

    ds_lidar_cut = ds_lidar.sel(time=slice(t_start, t_end))
    lidar_idx = ds_lidar_cut['time']

    ds_ptu = df_ptu.to_xarray()
    ds_ptu = ds_ptu.set_coords("Time")
    ds_ptu = ds_ptu.swap_dims({"index": "Time"})
    _, index = np.unique(ds_ptu["Time"], return_index=True)
    ds_ptu = ds_ptu.isel(Time=index)
    ds_ptu = ds_ptu.drop_isel(Time=-1)
    ds_ptu = ds_ptu.rename({'Time': 'time'})

    ds_ptu_interp = ds_ptu.interp(time=lidar_idx)

    # --- convert WXT time and filter ---
    datetime_strs = [f"{date_str} {t}" for t in df_wxt['TIME']]
    df_wxt['time_combined'] = pd.to_datetime(datetime_strs)
    df_wxt['time_combined'] = df_wxt['time_combined'] - pd.Timedelta(hours=1)

    mask = (df_wxt['time_combined'] >= t_start) & (df_wxt['time_combined'] <= t_end)
    df_wxt_filtered = df_wxt.loc[mask]

    ds_wxt_filtered = df_wxt_filtered.to_xarray()
    ds_wxt_filtered = ds_wxt_filtered.set_coords("time_combined")
    ds_wxt_filtered = ds_wxt_filtered.swap_dims({"index": "time_combined"})
    ds_wxt_filtered = ds_wxt_filtered.rename({'time_combined': 'time'})

    ds_wxt_interp = ds_wxt_filtered.interp(time=lidar_idx)

    # --- calculate altitude with continuous reference pressure ---
    altitude_cont = height_from_pressure(
        np.float64(ds_ptu_interp['Pressure'][:]),
        p1=ds_wxt_interp['PTB_P'][:]
    )

    # --- save to CSV ---
    df_alt_cont = pd.DataFrame({
        "time": np.array(lidar_idx),
        "altitude": np.array(altitude_cont) * 100  # convert to cm?
    })

    df_alt_cont.to_csv(output_csv, index=False)
    print(f"Altitude data saved to {output_csv}")
