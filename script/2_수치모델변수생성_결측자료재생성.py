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

year=2020
for year in [2019,2020]:
    data_dir=f'{org_dir}/{year}'
    data_files = [i for i in os.listdir(data_dir) if i.endswith('csv')]
    for fn in data_files:
        print(f'{data_dir}/{fn}')
        #file load
        if year==2021:
            df_tp=pd.read_csv(rf'..\output\numeric_concat/test/{fn}',encoding='cp949')
        else:
            df_tp=pd.read_csv(rf'..\output\numeric_concat/train/{fn}',encoding='cp949')

        ##df_tp자료 idw를 통해 수치모델 변수 생성
        na_times=df_tp[df_tp['u_v1'].isnull()].times
        for df_time in na_times:
            print(f'{df_time}')
            o=df_tp.loc[df_tp.times==df_time,['Longitude','Latitude']]
     
             #hycom_v1
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_v1):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='1h'),
                        pd.to_datetime(df_time).ceil(freq='1h')])
                    if all(idx>dt_hycom_v1.min()):
                        idx=np.where([i in idx for i in dt_hycom_v1])[0]
    
                        if len(idx)!=2:
                            t0=1;t1=1
                            #floor가 결측인 경우/ceilling이 결측인 경우
                            if not pd.to_datetime(df_time).floor(freq='1h') in dt_hycom_v1:
                                floor_time=pd.to_datetime(df_time).floor(freq='1h')
                                while not floor_time in dt_hycom_v1:
                                    floor_time=floor_time-pd.to_timedelta(1,unit='h')
                                    t0=t0+2
                            if not pd.to_datetime(df_time).ceil(freq='1h') in dt_hycom_v1:
                                ceilling_time=pd.to_datetime(df_time).ceil(freq='1h')
                                
                                while not ceilling_time in dt_hycom_v1:
                                    ceilling_time=ceilling_time+pd.to_timedelta(1,unit='h')
                                    t1=t1+2
                            
                            idx=np.array([floor_time,  ceilling_time])
                            idx=np.where([i in idx for i in dt_hycom_v1])[0]
                            #선형보간 f_p= d2/(d1+d2)f_p1 + d1/(d1+d2)f_p2
                            u_v1_temp=u_v1[idx[0]]*(t1/(t0+t1))+u_v1[idx[1]]*(t0/(t0+t1))
                            v_v1_temp=v_v1[idx[0]]*(t1/(t0+t1))+v_v1[idx[1]]*(t0/(t0+t1))
                            ssh_temp=ssh[idx[0]]*(t1/(t0+t1))+ssh[idx[1]]*(t0/(t0+t1))
                        else:
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
                        raise
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
            df_tp.to_csv(rf'..\output\numeric_concat_수정/test/{fn}',index=False,encoding='cp949')
        else:
            df_tp.to_csv(rf'..\output\numeric_concat_수정/train/{fn}',index=False,encoding='cp949')
