# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:11:30 2021

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
o=np.load(f'{numeric_dir}/hycom_v2_depth_0.npz',allow_pickle=True)
lon3_d0=o['lon3'];lat3_d0=o['lat3'];dt_hycom_d0=o['dt_hycom'];u_d0=o['u'];v_d0=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_depth_2.npz',allow_pickle=True)
lon3_d2=o['lon3'];lat3_d2=o['lat3'];dt_hycom_d2=o['dt_hycom'];u_d2=o['u'];v_d2=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_depth_4.npz',allow_pickle=True)
lon3_d4=o['lon3'];lat3_d4=o['lat3'];dt_hycom_d4=o['dt_hycom'];u_d4=o['u'];v_d4=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_depth_6.npz',allow_pickle=True)
lon3_d6=o['lon3'];lat3_d6=o['lat3'];dt_hycom_d6=o['dt_hycom'];u_d6=o['u'];v_d6=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_depth_8.npz',allow_pickle=True)
lon3_d8=o['lon3'];lat3_d8=o['lat3'];dt_hycom_d8=o['dt_hycom'];u_d8=o['u'];v_d8=o['v']

o=np.load(f'{numeric_dir}/hycom_v2_depth_10.npz',allow_pickle=True)
lon3_d10=o['lon3'];lat3_d10=o['lat3'];dt_hycom_d10=o['dt_hycom'];u_d10=o['u'];v_d10=o['v']

##특정 날짜 추출 예시
#date=pd.to_datetime('2016-01-01 10')
#temp =lonlatProj.export_date_ecmwf(date,wu,wv,lon1,lat1,dt_ecmwf)

#date=dt_hycom_v1[0]#df_time=dt_hycom[0]
#temp2=lonlatProj.export_date_hycom_v1(date,u_v1,v_v1,ssh,lon2,lat2,dt_hycom_v1)

#date=dt_hycom_d2[0]#df_time=dt_hycom[0]
#temp3=lonlatProj.export_date_hycom_v2(date,u_d0,v_d0,lon3_d0,lat3_d0,dt_hycom_d0)

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
                temp1=lonlatProj.export_date_ecmwf(
                              pd.to_datetime(df_time),wu,wv,lon1,lat1,dt_ecmwf)            
                df_tp.loc[df_tp.times==df_time,'wu']=\
                    lonlatProj.gstat_idw(
                        temp1['lon'],temp1['lat'],temp1['wu'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'wv']=\
                    lonlatProj.gstat_idw(
                        temp1['lon'],temp1['lat'],temp1['wv'],o.Longitude,o.Latitude)
                # for lag in range(1,7):
                #     temp1=lonlatProj.export_date_ecmwf(
                #         pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h'),wu,wv,lon1,lat1,dt_ecmwf)            
                #     df_tp.loc[df_tp.times==df_time,f'wu_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp1['lon'],temp1['lat'],temp1['wu'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'wv_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp1['lon'],temp1['lat'],temp1['wv'],o.Longitude,o.Latitude)                        
            except:
                df_tp.loc[df_tp.times==df_time,['wu','wv']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'wu_lag{lag}',f'wv_lag{lag}']]=np.nan

            #hycom_v1
            try:
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
                # for lag in range(1,7):
                #     temp2=lonlatProj.export_date_hycom_v1(
                #         pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h'),u_v1,v_v1,ssh,lon2,lat2,dt_hycom_v1)
                #     df_tp.loc[df_tp.times==df_time,f'u_v1_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp2['lon'],temp2['lat'],temp2['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_v1_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp2['lon'],temp2['lat'],temp2['v'],o.Longitude,o.Latitude)                        
                #     df_tp.loc[df_tp.times==df_time,f'ssh_v1_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp2['lon'],temp2['lat'],temp2['ssh'],o.Longitude,o.Latitude)                        

            except:
                df_tp.loc[df_tp.times==df_time,['u_v1','v_v1','ssh_v1']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,
                #               [f'u_v1_lag{lag}',f'v_v1_lag{lag}',f'ssh_v1_lag{lag}']]=np.nan

            #hycom_v2
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_d0):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='3h'),
                        pd.to_datetime(df_time).ceil(freq='3h')])
                    idx=np.where([i in idx for i in dt_hycom_d0])[0]
                    
                    u_d0_temp=np.concatenate((u_d0[idx,:,:],u_d0[idx,:,:]))
                    u_d0_temp[1:3]=np.nan
                    v_d0_temp=np.concatenate((v_d0[idx,:,:],v_d0[idx,:,:]))
                    v_d0_temp[1:3]=np.nan

                    def my_fun(a):
                        return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                    u_d0_temp[1:3]=np.apply_along_axis(my_fun,0,u_d0[idx,:,:])
                    v_d0_temp[1:3]=np.apply_along_axis(my_fun,0,v_d0[idx,:,:])

                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d0_temp, v_d0_temp, lon3_d0, lat3_d0,                     
                        pd.date_range(pd.to_datetime(df_time).floor(freq='3h'),
                                      pd.to_datetime(df_time).ceil(freq='3h'),freq='1h'))
                else:
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d0, v_d0, lon3_d0, lat3_d0, dt_hycom_d0)
                #결측보간
                df_tp.loc[df_tp.times==df_time,'u_d0']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_d0']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
                
                # for lag in range(1,7):
                #     if (not pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h') in dt_hycom_d0):
                #         idx=np.array([
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h')])
                #         idx=np.where([i in idx for i in dt_hycom_d0])[0]
                        
                #         u_d0_temp=np.concatenate((u_d0[idx,:,:],u_d0[idx,:,:]))
                #         u_d0_temp[1:3]=np.nan
                #         v_d0_temp=np.concatenate((v_d0[idx,:,:],v_d0[idx,:,:]))
                #         v_d0_temp[1:3]=np.nan

                #         def my_fun(a):
                #             return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                #         u_d0_temp[1:3]=np.apply_along_axis(my_fun,0,u_d0[idx,:,:])
                #         v_d0_temp[1:3]=np.apply_along_axis(my_fun,0,v_d0[idx,:,:])

                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d0_temp, v_d0_temp, lon3_d0, lat3_d0,                     
                #             pd.date_range((pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #                           (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h'),freq='1h'))
                #     else:
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d0, v_d0, lon3_d0, lat3_d0, dt_hycom_d0)

                #     #결측보간
                #     df_tp.loc[df_tp.times==df_time,f'u_d0_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_d0_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)

            except:
                df_tp.loc[df_tp.times==df_time,['u_d0','v_d0']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'u_d0_lag{lag}',f'v_d0_lag{lag}']]=np.nan

            #depth2
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_d2):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='3h'),
                        pd.to_datetime(df_time).ceil(freq='3h')])
                    idx=np.where([i in idx for i in dt_hycom_d2])[0]
                    
                    u_d2_temp=np.concatenate((u_d2[idx,:,:],u_d2[idx,:,:]))
                    u_d2_temp[1:3]=np.nan
                    v_d2_temp=np.concatenate((v_d2[idx,:,:],v_d2[idx,:,:]))
                    v_d2_temp[1:3]=np.nan
            
                    def my_fun(a):
                        return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                    u_d2_temp[1:3]=np.apply_along_axis(my_fun,0,u_d2[idx,:,:])
                    v_d2_temp[1:3]=np.apply_along_axis(my_fun,0,v_d2[idx,:,:])
            
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d2_temp, v_d2_temp, lon3_d2, lat3_d2,                     
                        pd.date_range(pd.to_datetime(df_time).floor(freq='3h'),
                                      pd.to_datetime(df_time).ceil(freq='3h'),freq='1h'))
                else:
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d2, v_d2, lon3_d2, lat3_d2, dt_hycom_d2)
                #결측보간
                df_tp.loc[df_tp.times==df_time,'u_d2']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_d2']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
                
                # for lag in range(1,7):
                #     if (not pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h') in dt_hycom_d2):
                #         idx=np.array([
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h')])
                #         idx=np.where([i in idx for i in dt_hycom_d2])[0]
                        
                #         u_d2_temp=np.concatenate((u_d2[idx,:,:],u_d2[idx,:,:]))
                #         u_d2_temp[1:3]=np.nan
                #         v_d2_temp=np.concatenate((v_d2[idx,:,:],v_d2[idx,:,:]))
                #         v_d2_temp[1:3]=np.nan
            
                #         def my_fun(a):
                #             return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                #         u_d2_temp[1:3]=np.apply_along_axis(my_fun,0,u_d2[idx,:,:])
                #         v_d2_temp[1:3]=np.apply_along_axis(my_fun,0,v_d2[idx,:,:])
            
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d2_temp, v_d2_temp, lon3_d2, lat3_d2,                     
                #             pd.date_range((pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #                           (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h'),freq='1h'))
                #     else:
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d2, v_d2, lon3_d2, lat3_d2, dt_hycom_d2)
            
                #     #결측보간
                #     df_tp.loc[df_tp.times==df_time,f'u_d2_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_d2_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
            
            except:
                df_tp.loc[df_tp.times==df_time,['u_d2','v_d2']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'u_d2_lag{lag}',f'v_d2_lag{lag}']]=np.nan

            #depth4
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_d4):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='3h'),
                        pd.to_datetime(df_time).ceil(freq='3h')])
                    idx=np.where([i in idx for i in dt_hycom_d4])[0]
                    
                    u_d4_temp=np.concatenate((u_d4[idx,:,:],u_d4[idx,:,:]))
                    u_d4_temp[1:3]=np.nan
                    v_d4_temp=np.concatenate((v_d4[idx,:,:],v_d4[idx,:,:]))
                    v_d4_temp[1:3]=np.nan
            
                    def my_fun(a):
                        return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                    u_d4_temp[1:3]=np.apply_along_axis(my_fun,0,u_d4[idx,:,:])
                    v_d4_temp[1:3]=np.apply_along_axis(my_fun,0,v_d4[idx,:,:])
            
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d4_temp, v_d4_temp, lon3_d4, lat3_d4,                     
                        pd.date_range(pd.to_datetime(df_time).floor(freq='3h'),
                                      pd.to_datetime(df_time).ceil(freq='3h'),freq='1h'))
                else:
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d4, v_d4, lon3_d4, lat3_d4, dt_hycom_d4)
                #결측보간
                df_tp.loc[df_tp.times==df_time,'u_d4']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_d4']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
                
                # for lag in range(1,7):
                #     if (not pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h') in dt_hycom_d4):
                #         idx=np.array([
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h')])
                #         idx=np.where([i in idx for i in dt_hycom_d4])[0]
                        
                #         u_d4_temp=np.concatenate((u_d4[idx,:,:],u_d4[idx,:,:]))
                #         u_d4_temp[1:3]=np.nan
                #         v_d4_temp=np.concatenate((v_d4[idx,:,:],v_d4[idx,:,:]))
                #         v_d4_temp[1:3]=np.nan
            
                #         def my_fun(a):
                #             return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                #         u_d4_temp[1:3]=np.apply_along_axis(my_fun,0,u_d4[idx,:,:])
                #         v_d4_temp[1:3]=np.apply_along_axis(my_fun,0,v_d4[idx,:,:])
            
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d4_temp, v_d4_temp, lon3_d4, lat3_d4,                     
                #             pd.date_range((pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #                           (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h'),freq='1h'))
                #     else:
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d4, v_d4, lon3_d4, lat3_d4, dt_hycom_d4)
            
                #     #결측보간
                #     df_tp.loc[df_tp.times==df_time,f'u_d4_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_d4_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
            
            except:
                df_tp.loc[df_tp.times==df_time,['u_d4','v_d4']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'u_d4_lag{lag}',f'v_d4_lag{lag}']]=np.nan



            #depth6
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_d6):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='3h'),
                        pd.to_datetime(df_time).ceil(freq='3h')])
                    idx=np.where([i in idx for i in dt_hycom_d6])[0]
                    
                    u_d6_temp=np.concatenate((u_d6[idx,:,:],u_d6[idx,:,:]))
                    u_d6_temp[1:3]=np.nan
                    v_d6_temp=np.concatenate((v_d6[idx,:,:],v_d6[idx,:,:]))
                    v_d6_temp[1:3]=np.nan
            
                    def my_fun(a):
                        return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                    u_d6_temp[1:3]=np.apply_along_axis(my_fun,0,u_d6[idx,:,:])
                    v_d6_temp[1:3]=np.apply_along_axis(my_fun,0,v_d6[idx,:,:])
            
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d6_temp, v_d6_temp, lon3_d6, lat3_d6,                     
                        pd.date_range(pd.to_datetime(df_time).floor(freq='3h'),
                                      pd.to_datetime(df_time).ceil(freq='3h'),freq='1h'))
                else:
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d6, v_d6, lon3_d6, lat3_d6, dt_hycom_d6)
                #결측보간
                df_tp.loc[df_tp.times==df_time,'u_d6']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_d6']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
                
                # for lag in range(1,7):
                #     if (not pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h') in dt_hycom_d6):
                #         idx=np.array([
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h')])
                #         idx=np.where([i in idx for i in dt_hycom_d6])[0]
                        
                #         u_d6_temp=np.concatenate((u_d6[idx,:,:],u_d6[idx,:,:]))
                #         u_d6_temp[1:3]=np.nan
                #         v_d6_temp=np.concatenate((v_d6[idx,:,:],v_d6[idx,:,:]))
                #         v_d6_temp[1:3]=np.nan
            
                #         def my_fun(a):
                #             return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                #         u_d6_temp[1:3]=np.apply_along_axis(my_fun,0,u_d6[idx,:,:])
                #         v_d6_temp[1:3]=np.apply_along_axis(my_fun,0,v_d6[idx,:,:])
            
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d6_temp, v_d6_temp, lon3_d6, lat3_d6,                     
                #             pd.date_range((pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #                           (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h'),freq='1h'))
                #     else:
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d6, v_d6, lon3_d6, lat3_d6, dt_hycom_d6)
            
                #     #결측보간
                #     df_tp.loc[df_tp.times==df_time,f'u_d6_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_d6_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
            
            except:
                df_tp.loc[df_tp.times==df_time,['u_d6','v_d6']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'u_d6_lag{lag}',f'v_d6_lag{lag}']]=np.nan

            #depth8
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_d8):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='3h'),
                        pd.to_datetime(df_time).ceil(freq='3h')])
                    idx=np.where([i in idx for i in dt_hycom_d8])[0]
                    
                    u_d8_temp=np.concatenate((u_d8[idx,:,:],u_d8[idx,:,:]))
                    u_d8_temp[1:3]=np.nan
                    v_d8_temp=np.concatenate((v_d8[idx,:,:],v_d8[idx,:,:]))
                    v_d8_temp[1:3]=np.nan
            
                    def my_fun(a):
                        return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                    u_d8_temp[1:3]=np.apply_along_axis(my_fun,0,u_d8[idx,:,:])
                    v_d8_temp[1:3]=np.apply_along_axis(my_fun,0,v_d8[idx,:,:])
            
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d8_temp, v_d8_temp, lon3_d8, lat3_d8,                     
                        pd.date_range(pd.to_datetime(df_time).floor(freq='3h'),
                                      pd.to_datetime(df_time).ceil(freq='3h'),freq='1h'))
                else:
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d8, v_d8, lon3_d8, lat3_d8, dt_hycom_d8)
                #결측보간
                df_tp.loc[df_tp.times==df_time,'u_d8']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_d8']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
                
                # for lag in range(1,7):
                #     if (not pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h') in dt_hycom_d8):
                #         idx=np.array([
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h')])
                #         idx=np.where([i in idx for i in dt_hycom_d8])[0]
                        
                #         u_d8_temp=np.concatenate((u_d8[idx,:,:],u_d8[idx,:,:]))
                #         u_d8_temp[1:3]=np.nan
                #         v_d8_temp=np.concatenate((v_d8[idx,:,:],v_d8[idx,:,:]))
                #         v_d8_temp[1:3]=np.nan
            
                #         def my_fun(a):
                #             return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                #         u_d8_temp[1:3]=np.apply_along_axis(my_fun,0,u_d8[idx,:,:])
                #         v_d8_temp[1:3]=np.apply_along_axis(my_fun,0,v_d8[idx,:,:])
            
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d8_temp, v_d8_temp, lon3_d8, lat3_d8,                     
                #             pd.date_range((pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #                           (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h'),freq='1h'))
                #     else:
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d8, v_d8, lon3_d8, lat3_d8, dt_hycom_d8)
            
                #     #결측보간
                #     df_tp.loc[df_tp.times==df_time,f'u_d8_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_d8_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
            
            except:
                df_tp.loc[df_tp.times==df_time,['u_d8','v_d8']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'u_d8_lag{lag}',f'v_d8_lag{lag}']]=np.nan

            #depth10
            try:
                if (not pd.to_datetime(df_time) in dt_hycom_d10):
                    idx=np.array([
                        pd.to_datetime(df_time).floor(freq='3h'),
                        pd.to_datetime(df_time).ceil(freq='3h')])
                    idx=np.where([i in idx for i in dt_hycom_d10])[0]
                    
                    u_d10_temp=np.concatenate((u_d10[idx,:,:],u_d10[idx,:,:]))
                    u_d10_temp[1:3]=np.nan
                    v_d10_temp=np.concatenate((v_d10[idx,:,:],v_d10[idx,:,:]))
                    v_d10_temp[1:3]=np.nan
            
                    def my_fun(a):
                        return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                    u_d10_temp[1:3]=np.apply_along_axis(my_fun,0,u_d10[idx,:,:])
                    v_d10_temp[1:3]=np.apply_along_axis(my_fun,0,v_d10[idx,:,:])
            
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d10_temp, v_d10_temp, lon3_d10, lat3_d10,                     
                        pd.date_range(pd.to_datetime(df_time).floor(freq='3h'),
                                      pd.to_datetime(df_time).ceil(freq='3h'),freq='1h'))
                else:
                    temp3=lonlatProj.export_date_hycom_v2(
                        pd.to_datetime(df_time), u_d10, v_d10, lon3_d10, lat3_d10, dt_hycom_d10)
                #결측보간
                df_tp.loc[df_tp.times==df_time,'u_d10']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                df_tp.loc[df_tp.times==df_time,'v_d10']=\
                    lonlatProj.gstat_idw(
                        temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
                
                # for lag in range(1,7):
                #     if (not pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h') in dt_hycom_d10):
                #         idx=np.array([
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h')])
                #         idx=np.where([i in idx for i in dt_hycom_d10])[0]
                        
                #         u_d10_temp=np.concatenate((u_d10[idx,:,:],u_d10[idx,:,:]))
                #         u_d10_temp[1:3]=np.nan
                #         v_d10_temp=np.concatenate((v_d10[idx,:,:],v_d10[idx,:,:]))
                #         v_d10_temp[1:3]=np.nan
            
                #         def my_fun(a):
                #             return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
                #         u_d10_temp[1:3]=np.apply_along_axis(my_fun,0,u_d10[idx,:,:])
                #         v_d10_temp[1:3]=np.apply_along_axis(my_fun,0,v_d10[idx,:,:])
            
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d10_temp, v_d10_temp, lon3_d10, lat3_d10,                     
                #             pd.date_range((pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).floor(freq='3h'),
                #                           (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')).ceil(freq='3h'),freq='1h'))
                #     else:
                #         temp3=lonlatProj.export_date_hycom_v2(
                #             (pd.to_datetime(df_time)-pd.to_timedelta(lag,unit='h')), u_d10, v_d10, lon3_d10, lat3_d10, dt_hycom_d10)
            
                #     #결측보간
                #     df_tp.loc[df_tp.times==df_time,f'u_d10_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['u'],o.Longitude,o.Latitude)        
                #     df_tp.loc[df_tp.times==df_time,f'v_d10_lag{lag}']=\
                #         lonlatProj.gstat_idw(
                #             temp3['lon'],temp3['lat'],temp3['v'],o.Longitude,o.Latitude)
            
            except:
                df_tp.loc[df_tp.times==df_time,['u_d10','v_d10']]=np.nan
                # for lag in range(1,7):
                #     df_tp.loc[df_tp.times==df_time,[f'u_d10_lag{lag}',f'v_d10_lag{lag}']]=np.nan

        if year==2021:
            #경로 나중에 바꿔야 함
            df_tp.to_csv(rf'..\output\numeric_concat/test/{fn}',index=False,encoding='cp949')
        else:
            df_tp.to_csv(rf'..\output\numeric_concat/train/{fn}',index=False,encoding='cp949')
# 수치모델자료 해당 위치 lag : 12시간 