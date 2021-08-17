# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:11:30 2021

@author: cjcho
"""
import lonlatProj
import pandas as pd

#load ecmwf
ecmwf_dir=r'D:\numeric\ERA5'
lon1,lat1,dt_ecmwf,wu,wv=lonlatProj.read_ecmwf(ecmwf_dir=ecmwf_dir)

#load hycom1
hycom_dir=r'D:\numeric\HYCOM/ALL'
lon2,lat2,dt_hycom,u,v,ssh=lonlatProj.read_hycom(hycom_dir=hycom_dir)
##특정 날짜 추출
date=pd.to_datetime('2016-01-01 10')
temp=lonlatProj.export_date_ecmwf(date,wu,wv,lon1,lat1,dt_ecmwf)
date=dt_hycom[0]
temp2=lonlatProj.export_date_hycom(date,u,v,ssh,lon2,lat2,dt_hycom)

##df_tp와 합치기
for df_time in df_tp.times:
    o=df_tp.loc[df_tp.times==df_time,['Longitude','Latitude']]
    temp=lonlatProj.export_date(df_time,wu,wv,longitude,latitude,dt_ecmwf)
    df_tp.loc[df_tp.times==df_time,'wu']=\
        lonlatProj.gstat_idw(
            temp['lon'],temp['lat'],temp['wu'],o.Longitude,o.Latitude)

    df_tp.loc[df_tp.times==df_time,'wv']=\
        lonlatProj.gstat_idw(
            temp['lon'],temp['lat'],temp['wv'],o.Longitude,o.Latitude)


