# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:09:13 2021

@author: cjcho
"""

import lonlatProj
import pandas as pd
import os
import numpy as np
org_dir='../data/조사원/1_이상치제거'
numeric_dir='../data'
numeric_dir=r'D:\output'
#load ecmwf
o=np.load(f'{numeric_dir}/ecmwf.npz',allow_pickle=True)
lon1=o['lon1'];lat1=o['lat1'];dt_ecmwf=o['dt_ecmwf'];wu=o['wu'];wv=o['wv']

#load hycom1
o=np.load(f'{numeric_dir}/hycom_v1.npz',allow_pickle=True)
lon2=o['lon2'];lat2=o['lat2'];dt_hycom_v1=o['dt_hycom'];u_v1=o['u'];v_v1=o['v'];ssh=o['ssh']

#load hycom2
# o=np.load(f'{numeric_dir}/hycom_v2_depth_0.npz',allow_pickle=True)
# lon3_d0=o['lon3'];lat3_d0=o['lat3'];dt_hycom_d0=o['dt_hycom'];u_d0=o['u'];v_d0=o['v']

# o=np.load(f'{numeric_dir}/hycom_v2_depth_2.npz',allow_pickle=True)
# lon3_d2=o['lon3'];lat3_d2=o['lat3'];dt_hycom_d2=o['dt_hycom'];u_d2=o['u'];v_d2=o['v']

# o=np.load(f'{numeric_dir}/hycom_v2_depth_4.npz',allow_pickle=True)
# lon3_d4=o['lon3'];lat3_d4=o['lat3'];dt_hycom_d4=o['dt_hycom'];u_d4=o['u'];v_d4=o['v']

# o=np.load(f'{numeric_dir}/hycom_v2_depth_6.npz',allow_pickle=True)
# lon3_d6=o['lon3'];lat3_d6=o['lat3'];dt_hycom_d6=o['dt_hycom'];u_d6=o['u'];v_d6=o['v']

# o=np.load(f'{numeric_dir}/hycom_v2_depth_8.npz',allow_pickle=True)
# lon3_d8=o['lon3'];lat3_d8=o['lat3'];dt_hycom_d8=o['dt_hycom'];u_d8=o['u'];v_d8=o['v']

# o=np.load(f'{numeric_dir}/hycom_v2_depth_10.npz',allow_pickle=True)
# lon3_d10=o['lon3'];lat3_d10=o['lat3'];dt_hycom_d10=o['dt_hycom'];u_d10=o['u'];v_d10=o['v']

year=2020
for year in [2016,2017,2019,2020,2021]:
    data_dir=f'{org_dir}/{year}'
    data_files = [i for i in os.listdir(data_dir) if i.endswith('csv')]
    for fn in data_files:
        print(f'{data_dir}/{fn}')
        #file load
        if year==2021:
            df_tp=pd.read_csv(rf'..\output\create_vars/test/{fn}',encoding='cp949')
        else:
            df_tp=pd.read_csv(rf'..\output\create_vars/train/{fn}',encoding='cp949')

        ##df_tp자료 idw를 통해 수치모델 변수 생성
        for df_time in df_tp.times:
            print(f'{df_time}')
            o=df_tp.loc[df_tp.times==df_time,['Longitude','Latitude']]
            #ecmwf
            try:
                if (not pd.to_datetime(df_time) in dt_ecmwf):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='1h'),
                        pd.to_datetime(df_time).ceil(freq='1h')])
                    idx=np.where([i in idx for i in dt_ecmwf])[0]
                    
                    wu_temp=(wu[idx[0]]+wu[idx[1]])/2
                    wv_temp=(wv[idx[0]]+wv[idx[1]])/2
                    temp1=pd.DataFrame({'wu':wu_temp.flatten(),
                                  'wv':wv_temp.flatten(),
                                  'lon':np.array(lon1.tolist()*len(lat1)),
                                  'lat':np.array([np.repeat(i,len(lon1))
                                                  for i in lat1]).flatten()}
                                  )
                else:
                    temp1=lonlatProj.export_date_ecmwf(
                                          pd.to_datetime(df_time),wu,wv,lon1,lat1,dt_ecmwf)            
                #interpolation
                df_tp.loc[df_tp.times==df_time,'wu']=\
                    lonlatProj.gstat_idw(
                        temp1['lon'],temp1['lat'],temp1['wu'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'wv']=\
                    lonlatProj.gstat_idw(
                        temp1['lon'],temp1['lat'],temp1['wv'],o.Longitude,o.Latitude)
            except:
                df_tp.loc[df_tp.times==df_time,['wu','wv']]=np.nan
            
             #hycom_v1
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_v1):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='1h'),
                        pd.to_datetime(df_time).ceil(freq='1h')])
                    idx=np.where([i in idx for i in dt_hycom_v1])[0]
                    
                    u_v1_temp=(u_v1[idx[0]]+u_v1[idx[1]])/2
                    v_v1_temp=(v_v1[idx[0]]+v_v1[idx[1]])/2
                    ssh_temp=(ssh[idx[0]]+ssh[idx[1]])/2
                    
                    temp2=pd.DataFrame({'u':u_v1_temp.flatten(),
                                  'v':v_v1_temp.flatten(),
                                  'ssh':ssh_temp.flatten(),
                                  'lon':np.array(lon2.tolist()*len(lat2)),
                                  'lat':np.array([np.repeat(i,len(lon2))
                                                  for i in lat2]).flatten()}
                                  )
                else:
                    temp2=lonlatProj.export_date_hycom_v1(
                        pd.to_datetime(df_time), u_v1, v_v1, ssh, lon2, lat2, dt_hycom_v1)
                df_tp.loc[df_tp.times==df_time,'u_v1']=\
                    lonlatProj.gstat_idw(
                        temp2['lon'],temp2['lat'],temp2['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_v1']=\
                    lonlatProj.gstat_idw(
                        temp2['lon'],temp2['lat'],temp2['v'],o.Longitude,o.Latitude)
                df_tp.loc[df_tp.times==df_time,'ssh_v1']=\
                    lonlatProj.gstat_idw(
                        temp2['lon'],temp2['lat'],temp2['ssh'],o.Longitude,o.Latitude)
                    
            except:
                df_tp.loc[df_tp.times==df_time,['u_v1','v_v1','ssh_v1']]=np.nan
        if year==2021:
            #경로 나중에 바꿔야 함
            df_tp.to_csv(rf'..\output\numeric_concat/test/{fn}',index=False,encoding='cp949')
        else:
            df_tp.to_csv(rf'..\output\numeric_concat/train/{fn}',index=False,encoding='cp949')
