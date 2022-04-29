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
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

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
    lat=float(row[10])
    lon=float(row[11])

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

    lats = [lt1, lt2, lt3, lt4]
    lons = [ln1, ln2, ln3, ln4]    

    wsn = []
    sn = []
    uu = []
    vv = []
    pr = []
    t2 = []
    for lt, ln in zip(lats, lons):
      wsn1, sn1, uu1, vv1, pr1, t2_1 = getModelData(datai, dataf, lt, ln)
      wsn.append(wsn1)
      sn.append(sn1)
      uu.append(uu1)
      vv.append(vv1)
      pr.append(pr1)
      t2.append(t2_1)
    
    wsn = xr.combine_nested(wsn, concat_dim=['south_north']).mean(axis=0)
    sn = xr.combine_nested(sn, concat_dim=['south_north']).mean(axis=0)
    uu = xr.combine_nested(uu, concat_dim=['south_north']).mean(axis=0)
    vv = xr.combine_nested(vv, concat_dim=['south_north']).mean(axis=0)
    pr = xr.combine_nested(pr, concat_dim=['south_north']).mean(axis=0)
    t2 = xr.combine_nested(t2, concat_dim=['south_north']).mean(axis=0)
    ws = np.sqrt(np.power(uu,2) + np.power(vv,2))

    # Do stuff
    # Distribution of the wind, temperature, precipitation

    # wind
    bin_wind = np.arange(0,21,1)
    ws_hist, _ = np.histogram(ws, bin_wind)
    # temp
    bin_t2 = np.arange(-30,31,2)
    t2_hist, _ = np.histogram(t2, bin_t2)

    # precip
    bin_pr = np.arange(0,20,1)
    pr_hist, _ = np.histogram(pr, bin_pr)
    total_pr = np.sum(pr)
    
    # wsn
    bin_wsn = np.arange(0,20,1)
    wsn_hist, _ = np.histogram(wsn, bin_pr)
    total_wsn = np.sum(wsn)

    # sn
    bin_sn = np.arange(0,20,1)
    sn_hist, _ = np.histogram(sn, bin_pr)
    total_sn = np.sum(sn)

    # See the script that generate the nice plots, to generate them again. This time with the amount of snow.

    # Plot the histograms and the nice plots.
    # 2 x 4 image. 2 x 2 histogram; 2 x 2 the nice image
    fig, axs = plt.subplot_mosaic([['a)', 'b)'], ['c)', 'd)'], ['e)', 'e)'], ['e)', 'e)']],
                              constrained_layout=True, figsize=(10, 14))
    for label, ax in axs.items():
    # label physical distance in and down:
      trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
      ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='18', va='bottom', fontfamily='serif')

      plt.suptitle('Change the title here', fontsize=20)

      if label == 'a)':
        ax.bar(bin_wind[:-1], ws_hist)
        ax.set_title('Wind Distribution', fontsize=18)
      if label == 'b)':
        ax.bar(bin_t2[:-1], t2_hist)
        ax.set_title('Temperature Distribution', fontsize=18)
      if label == 'c)':
        ax.bar(bin_wsn[:-1], wsn_hist)
        ax.set_title('Wetsnow Distribution', fontsize=18)
      if label == 'd)':
        ax.bar(bin_sn[:-1], sn_hist)
        ax.set_title('Snow Distribution', fontsize=18)
      if label == 'e)':
        # plot the nice plot
        print('oi')
    
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
