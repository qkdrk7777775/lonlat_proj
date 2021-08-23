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


train_dir=r'../output/create_vars/train'
test_dir=r'../output/create_vars/test'
bound=30000

#train data set
tr_x=list();tr_target_x=list();tr_target_y=list()
for fn in os.listdir(train_dir):
    org_df=pd.read_csv(f'{train_dir}/{fn}',engine='python')
    #연속적으로 못만드는 변수 제거
    org_df=org_df.drop(['p_V_dist','p_U_dist'],axis=1)
    org_df[['del_lat','del_lon']]=org_df[
        ['Latitude','Longitude']].diff(periods=1)
    org_df[['del_x','del_y']]=org_df[['x','y']].diff(periods=1)
    
    org_df['del_x']=np.where(np.abs(org_df['del_x'])>bound,np.nan,org_df['del_x'])
    org_df['del_y']=np.where(np.abs(org_df['del_y'])>bound,np.nan,org_df['del_y'])
    #target 설정
    """
    target: pred_x-x, pred_y-y
    del_x = x-p_x
    del_x.shift(-1) = pred_x - x
    """
    org_df[['tar_del_x','tar_del_y']]=org_df[['del_x','del_y']].shift(-1)
    org_df=org_df.dropna()

    tr_x.append(org_df.drop(['times','tar_del_x','tar_del_y'],axis=1))
    tr_target_x.append(org_df['tar_del_x'])
    tr_target_y.append(org_df['tar_del_y'])

tr_x=pd.concat(tr_x)
tr_target_x=pd.concat(tr_target_x)
tr_target_y=pd.concat(tr_target_y)

#model train
lgb_x.fit(tr_x,tr_target_x)
lgb_y.fit(tr_x,tr_target_y)

rf_x.fit(tr_x,tr_target_x)
rf_y.fit(tr_x,tr_target_y)

et_x.fit(tr_x,tr_target_x)
et_y.fit(tr_x,tr_target_y)


#test data set
for fn in os.listdir(test_dir):
    org_df=pd.read_csv(f'{test_dir}/{fn}',engine='python')
    
    org_df=org_df.drop(['p_V_dist','p_U_dist'],axis=1)#연속적으로 못만드는 변수 제거
    org_df=org_df.loc[1:].reset_index().drop('index',axis=1)#index 초기화
    org_df[['del_lat','del_lon','del_x','del_y']]=np.nan#필요하지만 없는 변수 미리 생성
    org_df[['tar_lon','tar_lat']]=org_df[['Longitude','Latitude']]#검증용 target 미리 생성
    #1행을 제외한 결측자료 생성
    org_df.loc[1:,['Latitude', 'Longitude', 'p_lat', 'p_lon', 'degrees', 'x', 'y',
               'p_x', 'p_y', 'dist', 'U_dist', 'V_dist']]=np.nan
    #1행 모든 변수 만들기
    p_lat,p_lon,lat,lon=org_df.loc[0,['p_lat','p_lon','Latitude','Longitude']]
    org_df.loc[0,['degrees','x','y','p_x','p_y','dist','U_dist',\
              'V_dist','del_lat','del_lon','del_x','del_y']]=\
        lonlatProj.recracive_create_vars(p_lat,p_lon,lat,lon)
    
    nrow=org_df.shape[0]
    #recracive prediction
    for i in range(1,nrow):
        if i%10==0:
            print(i,nrow)
        
        org_df.loc[i,'p_x']=org_df.loc[i-1,'x']
        """
        target: pred_x-x, pred_y-y
        del_x = x-p_x
        del_x.shift(-1) = pred_x - x
        """
        #현시차 예측값 pred_x-x, 전시차 예측값 x
        org_df.loc[i,'x']=lgb_x.predict(
            org_df.drop(['times','tar_lon','tar_lat'],axis=1).loc[(i-1):(i-1)])+org_df['x'][i-1]
        
        org_df.loc[i,'p_y']=org_df.loc[i-1,'y']
        org_df.loc[i,'y']=lgb_y.predict(
            org_df.drop(['times','tar_lon','tar_lat'],axis=1).loc[(i-1):(i-1)])+org_df['y'][i-1]
        lon,lat=lonlatProj.tm_to_wgs84(org_df.loc[i,'x'], org_df.loc[i,'y'])
        p_lon,p_lat=lonlatProj.tm_to_wgs84(org_df.loc[i,'p_x'], org_df.loc[i,'p_y'])
        org_df.loc[i,['Longitude','Latitude','p_lon','p_lat']]=lon,lat,p_lon,p_lat
        org_df.loc[i,['degrees','x','y','p_x','p_y','dist','U_dist',\
                  'V_dist','del_lat','del_lon','del_x','del_y']]=\
            lonlatProj.recracive_create_vars(p_lat,p_lon,lat,lon)
    
#org_df.to_csv("Z:\cj\lonlat\output/test.csv",index=False)




