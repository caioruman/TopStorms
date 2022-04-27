import os
import pandas as pd
import numpy as np
import xarray as xr
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter
from metpy.units import units
from metpy.units import pandas_dataframe_to_unit_arrays
import metpy.calc as mpcalc
from glob import glob

def main():

  '''
  1) Read Excel, select event.
  2) From the location of the event, open the dataframe with the EC data
  3) Open the model data and extract data from the event
  '''

  stormsFile = 'classification_event_nb_power.xlsx'

  df = pd.read_excel(stormsFile, skiprows=[1])

  aux1 = xr.open_dataset('/chinook/marinier/CONUS_2D/CTRL/2000/wrf2d_d01_CTRL_T2_200010-200012.nc', engine='netcdf4') 

  # resolution of the simulation in km  
  km = 4

  # go throught each row. Compute the total snow precipitation (and wet snow), get the wind and temperature.
  # get the initial data and 48h after it (or until precip < 0.1).
  for row in df.itertuples():
    
    datai = datetime(row[1], row[2], row[3], 0)
    dataf = datai + relativedelta(hours=48)
    lat=row[10]
    lon=row[11]

    # check if the string do not have snow in it (lowercase)
    if not 'snow' in row[4].lower():
      continue

    # get the 4 pair of points around the location
    f_lt = km / 111.3
    f_ln = km / 111.3 / np.cos(np.deg2rad(lat))
    lt1 = lat - f_lt/2
    ln1 = lon - f_ln/2
    lt2 = lat + f_lt/2
    ln2 = lon - f_ln/2
    lt3 = lat - f_lt/2
    ln3 = lon + f_ln/2
    lt4 = lat + f_lt/2
    ln4 = lon + f_ln/2    

    wsn, sn, uu, vv, pr, t2 = getModelData(datai, dataf, lt1, lt2)
    
    
    sys.exit()

# Open the model and extract data
def getModelData(datai, dataf, lat, lon):
  store = '/chinook/cruman/Data/WetSnow'
  st = '/chinook/marinier/CONUS_2D/CTRL/'
  if 1 <= datai.month <= 3:
    mi = 1
    mf = 3
  elif 4 <= datai.month <= 6:
    mi = 4
    mf = 6
  elif 7 <= datai.month <= 9:
    mi = 7
    mf = 9
  else:
    mi = 10
    mf = 12
  wsn = xr.open_dataset(f'{store}/WetSnow_SN_{datai.year}{mi:02d}-{datai.year}{mf:02d}.nc', engine='netcdf4')        
  i, j = geo_idx([lat, lon], np.array([wsn.XLAT, wsn.XLONG]))  

  wsn = wsn.SN_2C[:,i,j]
  
  sn = xr.open_dataset(f'{st}/{datai.year}/wrf2d_d01_CTRL_SNOW_ACC_NC_{datai.year}{mi:02d}-{datai.year}{mf:02d}.nc', engine='netcdf4')  
  i, j = geo_idx([lat, lon], np.array([sn.XLAT, sn.XLONG]))

  sn = sn.SNOW_ACC_NC[:,i,j]
  uu = xr.open_dataset(f'{st}/{datai.year}/wrf2d_d01_CTRL_EU10_{datai.year}{mi:02d}-{datai.year}{mf:02d}.nc', engine='netcdf4')    
  uu = uu.EU10[:,i,j]
  vv = xr.open_dataset(f'{st}/{datai.year}/wrf2d_d01_CTRL_EV10_{datai.year}{mi:02d}-{datai.year}{mf:02d}.nc', engine='netcdf4')
  vv = vv.EV10[:,i,j]
  pr = xr.open_dataset(f'{st}/{datai.year}/wrf2d_d01_CTRL_PREC_ACC_NC_{datai.year}{mi:02d}-{datai.year}{mf:02d}.nc', engine='netcdf4')
  pr = pr.PREC_ACC_NC[:,i,j]
  t2 = xr.open_dataset(f'{st}/{datai.year}/wrf2d_d01_CTRL_T2_{datai.year}{mi:02d}-{datai.year}{mf:02d}.nc', engine='netcdf4')
  t2 = t2.T2[:,i,j]        
  
  if (datai.month != dataf.month) and (datai.month % 3 == 0):
    #open the second dataset and attach to the first        
    print('oi')
    if 1 <= dataf.month <= 3:
      mif = 1
      mff = 3
    elif 4 <= dataf.month <= 6:
      mif = 4
      mff = 6
    elif 7 <= dataf.month <= 9:
      mif = 7
      mff = 9
    else:
      mif = 10
      mff = 12
    wsnf = xr.open_dataset(f'{store}/WetSnow_SN_{dataf.year}{mif:02d}-{dataf.year}{mff:02d}.nc', engine='netcdf4')        
    i, j = geo_idx([lat, lon], np.array([wsnf.XLAT, wsnf.XLONG]))
    
    print("open wsnf")
    wsnf = wsnf.SN_2C[:,i,j]

    snf = xr.open_dataset(f'{st}/{dataf.year}/wrf2d_d01_CTRL_SNOW_ACC_NC_{dataf.year}{mif:02d}-{dataf.year}{mff:02d}.nc', engine='netcdf4')
    i, j = geo_idx([lat, lon], np.array([snf.XLAT, snf.XLONG]))
    
    print("open other data")
    snf = snf.SNOW_ACC_NC[:,i,j]
    uuf = xr.open_dataset(f'{st}/{dataf.year}/wrf2d_d01_CTRL_EU10_{dataf.year}{mif:02d}-{dataf.year}{mff:02d}.nc', engine='netcdf4')    
    uuf = uuf.EU10[:,i,j]
    vvf = xr.open_dataset(f'{st}/{dataf.year}/wrf2d_d01_CTRL_EV10_{dataf.year}{mif:02d}-{dataf.year}{mff:02d}.nc', engine='netcdf4')
    vvf = vvf.EV10[:,i,j]
    prf = xr.open_dataset(f'{st}/{dataf.year}/wrf2d_d01_CTRL_PREC_ACC_NC_{dataf.year}{mif:02d}-{dataf.year}{mff:02d}.nc', engine='netcdf4')
    prf = prf.PREC_ACC_NC[:,i,j]
    t2f = xr.open_dataset(f'{st}/{dataf.year}/wrf2d_d01_CTRL_T2_{dataf.year}{mif:02d}-{dataf.year}{mff:02d}.nc', engine='netcdf4')
    t2f = t2f.T2[:,i,j]        
    
    print("concatenando data")
    wsn = xr.concat([wsn, wsnf], dim='Time')
    print("wsn done")
    sn = xr.concat([sn, snf], dim='Time')
    print('sn done')
    uu = xr.concat([uu, uuf], dim='Time')
    print('uu done')
    vv = xr.concat([vv, vvf], dim='Time')
    print('vv done')
    pr = xr.concat([pr, prf], dim='Time')
    print('pr done')
    t2 = xr.concat([t2, t2f], dim='Time')
    print('t2 done')
    
      
  wsn = wsn.sel(Time=slice(datai.strftime('%Y-%m-%d %H:%M'), dataf.strftime('%Y-%m-%d %H:%M')))
  sn = sn.sel(Time=slice(datai.strftime('%Y-%m-%d %H:%M'), dataf.strftime('%Y-%m-%d %H:%M')))
  uu = uu.sel(Time=slice(datai.strftime('%Y-%m-%d %H:%M'), dataf.strftime('%Y-%m-%d %H:%M')))
  vv = vv.sel(Time=slice(datai.strftime('%Y-%m-%d %H:%M'), dataf.strftime('%Y-%m-%d %H:%M')))
  pr = pr.sel(Time=slice(datai.strftime('%Y-%m-%d %H:%M'), dataf.strftime('%Y-%m-%d %H:%M')))
  t2 = t2.sel(Time=slice(datai.strftime('%Y-%m-%d %H:%M'), dataf.strftime('%Y-%m-%d %H:%M')))
  
  return wsn, sn, uu, vv, pr, t2
  
  

def geo_idx(dd, dd_array, type="lat"):
  '''
    search for nearest decimal degree in an array of decimal degrees and return the index.
    np.argmin returns the indices of minium value along an axis.
    so subtract dd from all values in dd_array, take absolute value and find index of minimum.
    
    Differentiate between 2-D and 1-D lat/lon arrays.
    for 2-D arrays, should receive values in this format: dd=[lat, lon], dd_array=[lats2d,lons2d]
  '''
  if type == "lon" and len(dd_array.shape) == 1:
    dd_array = np.where(dd_array <= 180, dd_array, dd_array - 360)

  if (len(dd_array.shape) < 2):
    geo_idx = (np.abs(dd_array - dd)).argmin()
  else:
    if (dd_array[1] < 0).any():
      dd_array[1] = np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360)

    a = abs( dd_array[0]-dd[0] ) + abs(  np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360) - dd[1] )
    i,j = np.unravel_index(a.argmin(), a.shape)
    geo_idx = [i,j]

  return geo_idx

if __name__ == '__main__':
  main()