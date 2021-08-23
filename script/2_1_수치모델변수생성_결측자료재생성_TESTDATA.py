# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:41:22 2021

@author: cjcho
"""


import lonlatProj
import pandas as pd
import os
import numpy as np
from scipy import interpolate
import re
org_dir='../data/조사원/1_이상치제거'
numeric_dir='../data'
numeric_dir=r'D:\output'
#load ecmwf
o=np.load(f'{numeric_dir}/ecmwf.npz',allow_pickle=True)
lon1=o['lon1'];lat1=o['lat1'];dt_ecmwf=o['dt_ecmwf'];wu=o['wu'];wv=o['wv']

#load hycom1
o=np.load(f'{numeric_dir}/hycom_v1_2021.npz',allow_pickle=True)
lon2=o['lon2'];lat2=o['lat2'];dt_hycom_v1=o['dt_hycom'];u_v1=o['u'];v_v1=o['v'];ssh=o['ssh']

#load hycom2
o=np.load(f'{numeric_dir}/hycom_v2_year_2021_depth_0.npz',allow_pickle=True)
lon3_d0=o['lon3'];lat3_d0=o['lat3'];dt_hycom_d0=o['dt_hycom'];u_d0=o['u'];v_d0=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_year_2021_depth_2.npz',allow_pickle=True)
lon3_d2=o['lon3'];lat3_d2=o['lat3'];dt_hycom_d2=o['dt_hycom'];u_d2=o['u'];v_d2=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_year_2021_depth_4.npz',allow_pickle=True)
lon3_d4=o['lon3'];lat3_d4=o['lat3'];dt_hycom_d4=o['dt_hycom'];u_d4=o['u'];v_d4=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_year_2021_depth_6.npz',allow_pickle=True)
lon3_d6=o['lon3'];lat3_d6=o['lat3'];dt_hycom_d6=o['dt_hycom'];u_d6=o['u'];v_d6=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_year_2021_depth_8.npz',allow_pickle=True)
lon3_d8=o['lon3'];lat3_d8=o['lat3'];dt_hycom_d8=o['dt_hycom'];u_d8=o['u'];v_d8=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_year_2021_depth_10.npz',allow_pickle=True)
lon3_d10=o['lon3'];lat3_d10=o['lat3'];dt_hycom_d10=o['dt_hycom'];u_d10=o['u'];v_d10=o['v']

year=2020
wind_num=6
for year in [2021]:
    data_dir=f'{org_dir}/{year}'
    data_files = [i for i in os.listdir(data_dir) if i.endswith('csv')]
    for fn in data_files:
        print(f'{data_dir}/{fn}')
        #file load
        if year==2021:
            df_tp=pd.read_csv(rf'..\output\기존결과/test/{fn}',encoding='cp949')
        else:
            df_tp=pd.read_csv(rf'..\output\기존결과/train/{fn}',encoding='cp949')
        df_tp['times']=pd.to_datetime(df_tp['times'])
        # df_time=pd.to_datetime('2020-08-11 20:00')

        for df_time in df_tp.times:
            print(df_time)
            o=df_tp.loc[df_tp.times==pd.to_datetime(df_time)]
            for j in range(wind_num):
                #ecmwf
                try:
                    if (not pd.to_datetime(df_time) in dt_ecmwf):
                        idx=np.array([
                            pd.to_datetime(df_time).floor(freq='1h'),
                            pd.to_datetime(df_time).ceil(freq='1h')])
                        idx=np.where([i in idx for i in dt_ecmwf])[0]-j
                        
                        wu_temp=(wu[idx[0]]+wu[idx[1]])/2
                        wv_temp=(wv[idx[0]]+wv[idx[1]])/2
                    else:
                        idx=np.where(dt_ecmwf==pd.to_datetime(df_time))[0]-j
                        wu_temp=wu[idx[0]]
                        wv_temp=wv[idx[0]]
                    f_wu = interpolate.interp2d(lon1, lat1, wu_temp, kind='linear')
                    f_wv = interpolate.interp2d(lon1, lat1, wv_temp, kind='linear')
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'wind_u{j}']=f_wu(o['Longitude'],o['Latitude'])[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'wind_v{j}']=f_wv(o['Longitude'],o['Latitude'])[0]
                except:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'wind_u{j}',f'wind_v{j}']]=np.nan
                    
        na_times=df_tp.loc[df_tp[[i for i in df_tp.columns if re.compile('hycom').findall(i)]].isnull().any(axis=1),'times']
        for df_time in na_times:
            print(df_time)
            o=df_tp.loc[df_tp.times==pd.to_datetime(df_time)]
            for j in range(wind_num):
                #hycom
                dt_hycom0 = pd.date_range(dt_hycom_v1[0], dt_hycom_v1[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_v1))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat2,lon2),u_v1)
                f_v=interpolate.RegularGridInterpolator((x,lat2,lon2),v_v1)
                f_ssh=interpolate.RegularGridInterpolator((x,lat2,lon2),ssh)
                dt_idx=np.where(dt_hycom0==df_time)[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom1_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom1_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom1_ssh{j}']=\
                        f_ssh(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom1_u{j}',f'hycom1_v{j}',f'hycom1_ssh{j}']]=np.nan

                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom1_u{j}',f'hycom1_v{j}',f'hycom1_ssh{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat2,lon2),u_v1,method='nearest')
                    f_v=interpolate.RegularGridInterpolator((x,lat2,lon2),v_v1,method='nearest')
                    f_ssh=interpolate.RegularGridInterpolator((x,lat2,lon2),ssh,method='nearest')
                    dt_idx=np.where(dt_hycom0==df_time)[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom1_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom1_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom1_ssh{j}']=\
                            f_ssh(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom1_u{j}',f'hycom1_v{j}',f'hycom1_ssh{j}']]=np.nan

                #hycom0
                dt_hycom0 = pd.date_range(dt_hycom_d0[0], dt_hycom_d0[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_d0))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat3_d0,lon3_d0),u_d0)
                f_v=interpolate.RegularGridInterpolator((x,lat3_d0,lon3_d0),v_d0)
                
                dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d0_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d0_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom2_d0_u{j}',f'hycom2_d0_v{j}']]=np.nan
                    
                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom2_d0_u{j}',f'hycom2_d0_v{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat3_d0,lon3_d0),u_d0,method='nearest')
                    f_v=interpolate.RegularGridInterpolator((x,lat3_d0,lon3_d0),v_d0,method='nearest')
                    
                    dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d0_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d0_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        if np.isnan(f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]):
                            raise
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom2_d0_u{j}',f'hycom2_d0_v{j}']]=np.nan

                #depth2
                dt_hycom0 = pd.date_range(dt_hycom_d2[0], dt_hycom_d2[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_d2))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat3_d2,lon3_d2),u_d2)
                f_v=interpolate.RegularGridInterpolator((x,lat3_d2,lon3_d2),v_d2)
                
                dt_idx=np.where(dt_hycom0==df_time)[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d2_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d2_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom2_d2_u{j}',f'hycom2_d2_v{j}']]=np.nan
                    
                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom2_d2_u{j}',f'hycom2_d2_v{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat3_d2,lon3_d2),u_d2,method='nearest')
                    f_v=interpolate.RegularGridInterpolator((x,lat3_d2,lon3_d2),v_d2,method='nearest')
                    
                    dt_idx=np.where(dt_hycom0==df_time)[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d2_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d2_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom2_d2_u{j}',f'hycom2_d2_v{j}']]=np.nan

                #depth4
                dt_hycom0 = pd.date_range(dt_hycom_d4[0], dt_hycom_d4[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_d4))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat3_d4,lon3_d4),u_d4)
                f_v=interpolate.RegularGridInterpolator((x,lat3_d4,lon3_d4),v_d4)
                
                dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d4_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d4_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom2_d4_u{j}',f'hycom2_d4_v{j}']]=np.nan
                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom2_d4_u{j}',f'hycom2_d4_v{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat3_d4,lon3_d4),u_d4,method='nearest')
                    f_v=interpolate.RegularGridInterpolator((x,lat3_d4,lon3_d4),v_d4,method='nearest')
                    
                    dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d4_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d4_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom2_d4_u{j}',f'hycom2_d4_v{j}']]=np.nan

                #depth6
                dt_hycom0 = pd.date_range(dt_hycom_d6[0], dt_hycom_d6[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_d6))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat3_d6,lon3_d6),u_d6)
                f_v=interpolate.RegularGridInterpolator((x,lat3_d6,lon3_d6),v_d6)
                
                dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d6_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d6_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom2_d6_u{j}',f'hycom2_d6_v{j}']]=np.nan
                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom2_d6_u{j}',f'hycom2_d6_v{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat3_d6,lon3_d6),u_d6,method='nearest')   
                    f_v=interpolate.RegularGridInterpolator((x,lat3_d6,lon3_d6),v_d6,method='nearest')
                    
                    dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d6_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d6_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom2_d6_u{j}',f'hycom2_d6_v{j}']]=np.nan
                    
                #depth8
                dt_hycom0 = pd.date_range(dt_hycom_d8[0], dt_hycom_d8[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_d8))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat3_d8,lon3_d8),u_d8)
                f_v=interpolate.RegularGridInterpolator((x,lat3_d8,lon3_d8),v_d8)
                
                dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d8_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d8_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom2_d8_u{j}',f'hycom2_d8_v{j}']]=np.nan
                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom2_d8_u{j}',f'hycom2_d8_v{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat3_d8,lon3_d8),u_d8,method='nearest')   
                    f_v=interpolate.RegularGridInterpolator((x,lat3_d8,lon3_d8),v_d8,method='nearest')   
                    
                    dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d8_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d8_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom2_d8_u{j}',f'hycom2_d8_v{j}']]=np.nan

                #depth10
                dt_hycom0 = pd.date_range(dt_hycom_d10[0], dt_hycom_d10[-1], freq='30min')
                x=np.where(dt_hycom0.isin(dt_hycom_d10))[0]
                f_u=interpolate.RegularGridInterpolator((x,lat3_d10,lon3_d10),u_d10)
                f_v=interpolate.RegularGridInterpolator((x,lat3_d10,lon3_d10),v_d10)
                
                dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                if len(dt_idx)!=0:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d10_u{j}']=\
                        f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d10_v{j}']=\
                        f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                else:
                    df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                              [f'hycom2_d10_u{j}',f'hycom2_d10_v{j}']]=np.nan
                if any(df_tp.loc[df_tp.times==pd.to_datetime(df_time),[f'hycom2_d10_u{j}',f'hycom2_d10_v{j}']].isnull()):
                    f_u=interpolate.RegularGridInterpolator((x,lat3_d10,lon3_d10),u_d10,method='nearest')
                    f_v=interpolate.RegularGridInterpolator((x,lat3_d10,lon3_d10),v_d10,method='nearest')
                    
                    dt_idx=np.where(dt_hycom0==pd.to_datetime(df_time))[0]-j
                    if len(dt_idx)!=0:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d10_u{j}']=\
                            f_u(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),f'hycom2_d10_v{j}']=\
                            f_v(np.array([dt_idx,o['Latitude'],o['Longitude']]).T)[0]
                    else:
                        df_tp.loc[df_tp.times==pd.to_datetime(df_time),
                                  [f'hycom2_d10_u{j}',f'hycom2_d10_v{j}']]=np.nan

        if year==2021:
            #경로 나중에 바꿔야 함
            df_tp.to_csv(rf'..\output\기존결과_수정/test/{fn}',index=False,encoding='cp949')
        else:
            df_tp.to_csv(rf'..\output\기존결과_수정/train/{fn}',index=False,encoding='cp949')
