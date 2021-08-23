# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:46:37 2021

@author: cjcho
"""
#pip install lonlatProj
import lonlatProj
import pandas as pd
import numpy as np
import os
org_dir='../data/조사원/1_이상치제거'
for year in [2016,2017,2019,2020,2021]:
    """ 
    1. 1분단위 선형보간 
    2. 장기결측 (6시간 : na_time) 제거
    3. 변수 생성 :
        ['times': 관측시간, 
         'Latitude': 위도,
         'Longitude': 경도, 
         'p_lat': 전시차 위도, 
         'p_lon': 전시차 경도, 
         'degrees': 각도, 
         'x', 'y', 'p_x', 'p_y',
         'dist':전시차와 현재시차 거리, 
         'U_dist': dist*cos(degrees), 
         'V_dist': dist*sin(degrees)]
    """
    data_dir=f'{org_dir}/{year}'
    data_files = [i for i in os.listdir(data_dir) if i.endswith('csv')]
    print('year:',year)
    #if year==2021:
    #    limit_time=pd.Timedelta('0 days 01:00:00')
    ## 파일별로 변환
    for fn in data_files:
        print(f'{data_dir}/{fn}')
        org_df=pd.read_csv(f'{data_dir}/{fn}',engine='python',encoding='cp949')
        ### 변수명 처리
        if year==2021:
            df_tp=org_df[['관측시간', '위도','경도']]
        else:
            df_tp=org_df[['관측시간', '경도','위도']]
        df_tp.columns=['dt', 'Latitude', 'Longitude']        
        df_tp=df_tp.drop_duplicates('dt')
        
        df_tp['dt']=pd.to_datetime(df_tp['dt'])
        df_tp['dt']=df_tp.dt.round('1T')# 1분단위로 변경
        na_time=pd.Timedelta('0 days 06:00:00')
        
        df_tp['remove']=df_tp['dt'].diff()>na_time#장기결측 유무
        print(f'{str(na_time)} 시간 이상 결측자료 수 : {np.sum(df_tp.remove)}')
        df_tp=lonlatProj.contiTimeForm(df_tp, time_var_name='dt', freq='1T')#자료 1분간격 변환
        df_tp[['Latitude','Longitude']]=\
            df_tp[['Latitude','Longitude']].interpolate(method='values')#선형 보간
        
        df_tp['remove']=df_tp['remove'].interpolate(method='backfill')#장기 결측
        df_tp['Latitude']=np.where(df_tp['remove'],np.nan,df_tp['Latitude'])#장기결측인 자료 null값으로
        df_tp['Longitude']=np.where(df_tp['remove'],np.nan,df_tp['Longitude'])
        # df_tp=df_tp[~df_tp['remove']].drop('remove',axis=1)#장기결측이 아닌 자료 추출
        #ex) pd.DataFrame({'logit':[False,np.nan,np.nan,True,np.nan,False]})['logit'].interpolate(method='pad')
        df_tp=df_tp[[i in [0,30] for i in df_tp['times'].dt.minute.values]]# 1시간 간격자료 추출
        df_tp=df_tp.drop_duplicates('times')

        df_tp=lonlatProj.create_vars(df_tp)# 변수 생성
        print(df_tp.isna().sum())
        df_tp=df_tp.drop(['lat_rad','lon_rad','p_lat_rad','p_lon_rad'],axis=1)#불필요 변수 제거
        if year==2021:
            df_tp.to_csv(rf'..\output\create_vars/test/{fn}',index=False,encoding='cp949')
        else:
            df_tp.to_csv(rf'..\output\create_vars/train/{fn}',index=False,encoding='cp949')

