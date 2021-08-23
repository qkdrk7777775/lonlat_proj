# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:03:20 2021

@author: cjcho
"""
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import lonlatProj
method='linear'
model_save_path=r'../model_ml/test'
params_lgb = {
'boosting_type': 'gbdt',
'objective': 'regression',
'metric':'mse',
'n_estimators':200,
'max_depth':7, 
'random_state': 42
} #'num_threads' : -1

params_rf={
'max_depth':15,   
'random_state':42,
'n_jobs':-1
}

params_et={
'n_estimators':200,
'max_depth':15,   
'random_state':42,
'n_jobs':-1
}


lgb_x=lgb.LGBMRegressor(**params_lgb)
lgb_y=lgb.LGBMRegressor(**params_lgb)
rf_x=RandomForestRegressor(**params_rf)
rf_y=RandomForestRegressor(**params_rf)
et_x=ExtraTreesRegressor(**params_et)
et_y=ExtraTreesRegressor(**params_et)
    
numeric_dir=r'D:\output'
#load ecmwf
o=np.load(f'{numeric_dir}/ecmwf.npz',allow_pickle=True)
lon1=o['lon1'];lat1=o['lat1'];dt_ecmwf=o['dt_ecmwf'];wu=o['wu'];wv=o['wv']

#load hycom1
o=np.load(f'{numeric_dir}/hycom_v1.npz',allow_pickle=True)
lon2=o['lon2'];lat2=o['lat2'];dt_hycom_v1=o['dt_hycom'];u_v1=o['u'];v_v1=o['v'];ssh=o['ssh']

train_dir=r'../output/create_vars/train'
test_dir=r'../output/create_vars/test'
bound=30000

pred_hour=1
tr_x=list();tr_target_x=list();tr_target_y=list()
for fn in os.listdir(train_dir):
    org_df=pd.read_csv(f'{train_dir}/{fn}',engine='python')
    #target 설정
    """
    target: pred_x-x, pred_y-y
    """
    org_df['tar_del_x']=org_df['x'].shift(-pred_hour*2)-org_df['x']
    org_df['tar_del_y']=org_df['y'].shift(-pred_hour*2)-org_df['y']

    org_df[['del_lat','del_lon']]=org_df[
        ['Latitude','Longitude']].diff(periods=1)
    #수치모델 변수 생성
    lonlatProj.pred_intp_ecmwf(org_df,dt_ecmwf,wu,wv,lat1,lon1)
    lonlatProj.pred_intp_hycom(org_df,dt_ecmwf,wu,wv,lat1,lon1)

    org_df[['del_x','del_y']]=org_df[['x','y']].diff(periods=1)
    #이상자료 제거 
    org_df['del_x']=np.where(
        np.abs(org_df['del_x'])>bound,np.nan,org_df['del_x'])
    org_df['del_y']=np.where(
        np.abs(org_df['del_y'])>bound,np.nan,org_df['del_y'])
    
    #누적변화량
    org_df['cum_del_x']=np.abs(org_df['del_x']).rolling(window=6).sum()
    org_df['cum_del_y']=np.abs(org_df['del_y']).rolling(window=6).sum()
    #평균변화량
    org_df['mean_del_x']=org_df['del_x'].rolling(window=6).mean()
    org_df['mean_del_y']=org_df['del_y'].rolling(window=6).mean()

    #결측자료 제거 
    org_df=org_df.dropna()
    
    tr_x.append(org_df.drop(['times','tar_del_x','tar_del_y'],axis=1))
    tr_target_x.append(org_df['tar_del_x'])
    tr_target_y.append(org_df['tar_del_y'])

tr_x=pd.concat(tr_x)
tr_target_x=pd.concat(tr_target_x)
tr_target_y=pd.concat(tr_target_y)

#모델 fit
lgb_x.fit(tr_x,tr_target_x)
lgb_y.fit(tr_x,tr_target_y)

# rf_x.fit(tr_x,tr_target_x)
# rf_y.fit(tr_x,tr_target_y)

# et_x.fit(tr_x,tr_target_x)
# et_y.fit(tr_x,tr_target_y)

#검증 
for fn in os.listdir(test_dir):
    org_df=pd.read_csv(f'{test_dir}/{fn}',engine='python')
    #추가
    org_df[['Latitude','Longitude','p_lat','p_lon']]=\
        org_df[['Latitude','Longitude','p_lat','p_lon']].interpolate(method='values')
    
    org_df[2:]=lonlatProj.create_vars(org_df).drop(
        ['lat_rad','lon_rad','p_lat_rad','p_lon_rad'],axis=1)[2:]
    #end
    
    
    org_df[['del_lat','del_lon']]=org_df[
        ['Latitude','Longitude']].diff(periods=1)
    org_df[['del_x','del_y']]=org_df[['x','y']].diff(periods=1)#org_df['x']-org_df['p_x']
    
    org_df['del_x']=np.where(np.abs(org_df['del_x'])>bound,np.nan,org_df['del_x'])
    org_df['del_y']=np.where(np.abs(org_df['del_y'])>bound,np.nan,org_df['del_y'])
    #누적변화량
    org_df['cum_del_x']=np.abs(org_df['del_x']).rolling(window=6).sum()
    org_df['cum_del_y']=np.abs(org_df['del_y']).rolling(window=6).sum()
    #평균변화량
    org_df['mean_del_x']=org_df['del_x'].rolling(window=6).mean()
    org_df['mean_del_y']=org_df['del_y'].rolling(window=6).mean()
    #target 설정
    org_df[['tar_del_x','tar_del_y']]=org_df[['x','y']].shift(-pred_hour*2)
    
    # inter_cols=org_df.columns[[not i in ['times','tar_del_x','tar_del_y'] for i in org_df.columns]]
    # org_df[inter_cols]=org_df[inter_cols].interpolate(method='values')
    # org_df=org_df[org_df.drop(['tar_del_x','tar_del_y'],axis=1).notnull().any(1)]


    te_x=org_df.drop(['times','tar_del_x','tar_del_y'],axis=1)
    te_target_x=org_df['tar_del_x']
    te_target_y=org_df['tar_del_y']
    """
    target: pred_x-x, pred_y-y
    """
    pred_x=te_x['x']+lgb_x.predict(te_x)
    pred_y=te_x['y']+lgb_y.predict(te_x)
    
    out_file=org_df.copy()
    out_file=out_file[['times']]

    out_file[['pred_lon','pred_lat']]=pd.DataFrame(lonlatProj.tm_to_wgs84(pred_x, pred_y)).T
    out_file[['tar_lon','tar_lat']]=org_df[['Longitude','Latitude']].shift(-pred_hour*2)
    out_file.to_csv(rf'Z:\cj\lonlat\output/{fn}',index=False,encoding='cp949')

    #참고 사항

    #pred_x=pred_x[:101]
    #pred_y=pred_y[:101]
    mae=np.sum(
        np.sqrt(
            (pred_x-te_target_x)**2+(pred_y-te_target_y)**2)
        )/pred_x.shape[0]
    rmse=np.sqrt(
        np.sum(
            (pred_x-te_target_x)**2+(pred_y-te_target_y)**2)/pred_x.shape[0]
        )
    #np.sum(np.sqrt(pred_x-te_target_x)**2)/pred_x.shape[0]
    #np.sum(np.sqrt(pred_y-te_target_y)**2)/pred_y.shape[0]
    print(mae,rmse)




