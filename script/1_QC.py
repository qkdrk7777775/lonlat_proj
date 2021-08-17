# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:20:28 2021

@author: cjcho
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt
org_dir='../data/조사원/1_이상치제거'

year=2017#[2016,2017,2019,2020,2021]
data_dir=f'{org_dir}/{year}'
data_files = [i for i in os.listdir(data_dir) if i.endswith('csv')]
t=0

fn= data_files[t]
## 파일별로 변환
if year==2021:
    df_tp=pd.read_csv(rf'..\output\before_qc/test/{fn}',encoding='cp949')
else:
    df_tp=pd.read_csv(rf'..\output\before_qc/train/{fn}',encoding='cp949')
df_tp.times=pd.to_datetime(df_tp.times)

plt.cla()
plt.scatter(df_tp.U_dist,df_tp.V_dist)
plt.cla()
plt.scatter(df_tp.times,df_tp.dist)
plt.cla()
plt.scatter(df_tp.times,df_tp.Longitude)
plt.cla()
plt.scatter(df_tp.times,df_tp.Latitude)
plt.cla()
t=t+1

#300234064024020
import os
os.environ["PROJ_LIB"] = r'C:\Users\cjcho\anaconda3\envs\lonlat\Library\share'
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.cla()
map = Basemap(projection='merc', resolution = 'h',
              lat_0=32.35, lon_0=123.58,
              urcrnrlat=40, llcrnrlat=25, 
              llcrnrlon=120, urcrnrlon=145)
map.drawcoastlines()
map.fillcontinents()
import numpy as np
map.drawparallels(np.arange(-90,90,5),labels=[1,1,0,1], fontsize=8)
map.drawmeridians(np.arange(-180,180,5),labels=[1,1,0,1], rotation=45, fontsize=8)
x,y=map(df_tp.Longitude,df_tp.Latitude)
map.plot(x,y,'bo',markersize=.1)
plt.xlabel('Longitude', labelpad=40)
plt.ylabel('Latitude', labelpad=40)


#map.drawcountries()
#map.drawmapboundary()
