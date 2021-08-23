# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:06:39 2021

@author: cjcho
"""
import pyproj
import pandas as pd
import numpy as np
from netCDF4 import Dataset as NetCDFFile
import os
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
ro.r('install.packages(c("sp","gstat"))')

def gstat_idw(lon,lat,values,pred_lon,pred_lat):
    """
    use r package gstat idw    
    """
    r_f=ro.r('''
        library(sp)
        library(gstat)
         f<-function(lon,lat,values,pred_lon,pred_lat){
             pred=data.frame(lon=pred_lon,lat=pred_lat)
             coordinates(pred)=~lon+lat
             df=data.frame(lon,lat,values)
             df=df[complete.cases(df),]
             coordinates(df)=~lon+lat
    
         return(idw(values~1, df,pred)$var1.pred)
            }
         ''')
    return r_f(ro.FloatVector(lon),
               ro.FloatVector(lat),
               ro.FloatVector(values),
               ro.FloatVector(pred_lon),
               ro.FloatVector(pred_lat),
               )

def wgs84_to_tm(longitude, latitude):
    proj_lonlat = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
    proj_xy = pyproj.Proj('+proj=tmerc +lat_0=38 +lon_0=127.5 +k=0.9996 +x_0=1000000 +y_0=2000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
    return pyproj.transform( proj_lonlat, proj_xy, longitude, latitude )

def tm_to_wgs84(x, y):
    proj_lonlat = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
    proj_xy = pyproj.Proj('+proj=tmerc +lat_0=38 +lon_0=127.5 +k=0.9996 +x_0=1000000 +y_0=2000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
    return pyproj.transform( proj_xy, proj_lonlat, x, y )

def contiTimeForm(df,time_var_name,freq):
    df_=df.copy()
    df_=df_.set_index([time_var_name])
    dt_range=pd.date_range(min(df_.index),max(df_.index),freq=freq)
    na_date=set(dt_range)-set(df_.index)
    df_=pd.concat([df_,pd.DataFrame(columns=df_.columns,index=na_date)],axis=0)
    df_=df_.sort_index()
    df_=df_.reset_index().rename(columns={"index": "times"})
    return df_


def create_vars(df,R=6371000):
    """
    https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    """
    #create degrees
    df_=df.copy()
    df_[['p_lat','p_lon']]=df_[['Latitude','Longitude']].shift(1)
    df_[['lat_rad','lon_rad','p_lat_rad','p_lon_rad']]=df_[['Latitude', 'Longitude','p_lat','p_lon']]*np.pi/180
    del_lon=df_['lon_rad']-df_['p_lon_rad']
    
    x=np.cos(df_['lat_rad'])*np.sin(del_lon)
    y=np.cos(df_['p_lat_rad'])*np.sin(df_['lat_rad'])-np.sin(df_['p_lat_rad'])*np.cos(df_['lat_rad'])*np.cos(del_lon)
    #df_['degrees']=np.where(np.arctan(x/y)*180/np.pi>=0,np.arctan(x/y)*180/np.pi,360+np.arctan(x/y)*180/np.pi)
    df_['degrees']=np.where(np.arctan2(x,y)*180/np.pi>=0,np.arctan2(x,y)*180/np.pi,360+np.arctan2(x,y)*180/np.pi)
    #create distance
    df_['x'],df_['y']=wgs84_to_tm(df_['Longitude'].values,df_['Latitude'].values)
    df_[['p_x','p_y']]=df_[['x','y']].shift(periods=1)
    df_['dist']=np.sqrt(((df_['x']-df_['p_x'])**2)+((df_['y']-df_['p_y'])**2))
    df_['U_dist']=df_['dist']*np.cos(df_['degrees']*np.pi/180)
    df_['V_dist']=df_['dist']*np.sin(df_['degrees']*np.pi/180)
    df_[['p_U_dist','p_V_dist']]=df_[['U_dist','V_dist']].shift(periods=1)
    return df_

def recracive_create_vars(p_lat,p_lon,lat,lon,R=6371000):
    """
    https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    """
    #create degrees
    p_lat_rad=p_lat*np.pi/180
    p_lon_rad=p_lon*np.pi/180
    lat_rad=lat*np.pi/180
    lon_rad=lon*np.pi/180    
    del_lon=lon_rad-p_lon_rad
    
    x=np.cos(lat_rad)*np.sin(del_lon)
    y=np.cos(p_lat_rad)*np.sin(lat_rad)-np.sin(p_lat_rad)*np.cos(lat_rad)*np.cos(del_lon)
    degrees=float(np.where(np.arctan2(x,y)*180/np.pi>=0,np.arctan2(x,y)*180/np.pi,360+np.arctan2(x,y)*180/np.pi))
    #create distance
    x,y=wgs84_to_tm(lon,lat)
    p_x,p_y=wgs84_to_tm(p_lon,p_lat)
    dist=np.sqrt(((x-p_x)**2)+(y-p_y)**2)
    U_dist=dist*np.cos(degrees*np.pi/180)
    V_dist=dist*np.sin(degrees*np.pi/180)
    
    del_lon=lon-p_lon
    del_lat=lat-p_lat
    del_x=x-p_x
    del_y=y-p_y
    
    return degrees, x, y, p_x, p_y, dist, U_dist, V_dist, \
        del_lon, del_lat,del_x,del_y

def export_date_ecmwf(date,wu,wv,longitude,latitude,dt_ecmwf):
    idx=np.where(dt_ecmwf==date)
    return pd.DataFrame({'wu':wu[idx,:,:].flatten(),
                         'wv':wv[idx,:,:].flatten(),
                        'lon':np.array(longitude.tolist()*len(latitude)),
                        'lat':np.array([np.repeat(i,len(longitude))
                                         for i in latitude]).flatten()}
                       )

def export_date_hycom_v1(date,u,v,ssh,longitude,latitude,dt_hycom):
    idx=np.where(dt_hycom==date)
    return pd.DataFrame({'u':u[idx,:,:].flatten(),
                         'v':v[idx,:,:].flatten(),
                         'ssh':ssh[idx,:,:].flatten(),
                        'lon':np.array(longitude.tolist()*len(latitude)),
                        'lat':np.array([np.repeat(i,len(longitude))
                                         for i in latitude]).flatten()}
                       )

def export_date_hycom_v2(date,u,v,longitude,latitude,dt_hycom):
    idx=np.where(dt_hycom==date)
    return pd.DataFrame({'u':u[idx,:,:].flatten(),
                         'v':v[idx,:,:].flatten(),
                        'lon':np.array(longitude.tolist()*len(latitude)),
                        'lat':np.array([np.repeat(i,len(longitude))
                                         for i in latitude]).flatten()}
                       )

def read_ecmwf(ecmwf_dir):
    """
    ecmwfncfile concat
    longitude : ecmwf longitude
    latitude : ecmwf latitude
    wv: wave v component(m/s)
    wu: wave u componunt(m/s)
    """
    ecmwf_files = [i for i in os.listdir(ecmwf_dir) if i.endswith('nc')]
    ecmwf_files.sort()
    dates=list();wus=list();wvs=list()
    for fi,fn in enumerate(ecmwf_files):
        cnt_file = f'{ecmwf_dir}/{fn}'
        enc=NetCDFFile(cnt_file)
        if fi==0:
            for i in enc.variables:
                print(enc.variables[i])
            longitude=enc.variables['longitude'][:]
            latitude=enc.variables['latitude'][:]
        date=enc.variables['time'][:].data
        wu=enc.variables['u10'][:].data
        wv=enc.variables['v10'][:].data
        wu=np.where(wu==-32767,np.nan, wu)
        wv=np.where(wv==-32767,np.nan, wv)

        if len(wv.shape)==4:
            dates.append(date)
            wus.append(
                np.concatenate(
                    (wu[:2160,0,:,:],
                     wu[2160:,1,:,:])))
            wvs.append(
                np.concatenate(
                    (wv[:2160,0,:,:],
                     wv[2160:,1,:,:])))
        else:
            dates.append(date)
            wus.append(wu)
            wvs.append(wv)
        enc.close()
        
    dt_ecmwf=[pd.to_datetime('1900-01-01')+\
              pd.to_timedelta(int(h+9),unit='h') for h in np.concatenate(dates)]
    dt_ecmwf=np.array(dt_ecmwf).flatten()
    wv=np.concatenate(wvs)
    wu=np.concatenate(wus)
    return longitude,latitude,dt_ecmwf,wu,wv


def read_hycom_v1(hycom_dir):
    """
    hycom ncfile concat
    longitude : hycom longitude
    latitude : hycom latitude
    u : u component(m/s)
    v : v component(m/s)
    ssh : sea surface elevation(m)
    """
    hycom_files = [i for i in os.listdir(hycom_dir) if i.endswith('nc')]
    hycom_files.sort()
    us=list();vs=list();sshs=list();dates=list()
    for fi,fn in enumerate(hycom_files):
        cnt_file = f'{hycom_dir}/{fn}'
        enc=NetCDFFile(cnt_file)      
        if fi==0:
            for i in enc.variables:
                print(enc.variables[i])
            latitude=enc.variables['lat'][:].data
            longitude=enc.variables['lon'][:].data
        dt0=enc.variables['time'][:].data
        u=enc.variables['u_barotropic_velocity'][:].data
        v=enc.variables['v_barotropic_velocity'][:].data
        ssh=enc.variables['ssh'][:].data
        u=np.where(u==1.2676506002282294e+30,np.nan,u)
        v=np.where(v==1.2676506002282294e+30,np.nan,v)
        ssh=np.where(ssh==1.2676506002282294e+30,np.nan,ssh)
        dates.append(dt0)
        us.append(u)
        vs.append(v)
        sshs.append(ssh)
        enc.close()
    
    dt_hycom=[pd.to_datetime('2000-01-01')+\
              pd.to_timedelta(int(h+9),unit='h') for h in np.concatenate(dates).round(0)]
    dt_hycom=np.array(dt_hycom).flatten()
    u=np.concatenate(us)
    v=np.concatenate(vs)
    ssh=np.concatenate(sshs)
    return longitude,latitude,dt_hycom,u,v,ssh


def read_hycom_v2_1(hycom_dir,hycom_files,depth):
    """
    hycom ncfile concat
    longitude : hycom longitude
    latitude : hycom latitude
    depth : hycom depth
    u : u component(m/s)
    v : v component(m/s)
    """
    dates=list();us=list();vs=list()
    for fi,fn in enumerate(hycom_files):
        cnt_file = f'{hycom_dir}/{fn}'
        enc=NetCDFFile(cnt_file)      
        if fi==0:
            for i in enc.variables:
                print(enc.variables[i])
            latitude=enc.variables['lat'][:].data
            longitude=enc.variables['lon'][:].data
        dt0=enc.variables['time'][:].data
        u=(enc.variables['water_u'][:].data)[:,depth,:,:]
        v=(enc.variables['water_v'][:].data)[:,depth,:,:]
        u=np.where(u==-30000,np.nan,u)
        v=np.where(v==-30000,np.nan,v)
        us.append(u)
        vs.append(v)
        dates.append(dt0)
        enc.close()
        print(fn)
    dt_hycom=[pd.to_datetime('2000-01-01')+\
              pd.to_timedelta(int(h+9),unit='h') for h in np.concatenate(dates).round(0)]
    dt_hycom=np.array(dt_hycom).flatten()
    u=np.concatenate(us)
    v=np.concatenate(vs)
    return longitude,latitude,dt_hycom,u,v



def read_hycom_v2_2(hycom_dir,hycom_files,depth):
    """
    hycom ncfile concat
    longitude : hycom longitude
    latitude : hycom latitude
    depth : hycom depth
    u : u component(m/s)
    v : v component(m/s)
    """
    dates=list();
    us0=list();vs0=list()
    us2=list();vs2=list()
    us4=list();vs4=list()
    us6=list();vs6=list()
    us8=list();vs8=list()
    us10=list();vs10=list()

    for fi,fn in enumerate(hycom_files):
        cnt_file = f'{hycom_dir}/{fn}'
        enc=NetCDFFile(cnt_file)
        if fi==0:
            for i in enc.variables:
                print(enc.variables[i])
            latitude=enc.variables['lat'][:].data
            longitude=enc.variables['lon'][:].data
        dt0=enc.variables['time'][:].data
        u=(enc.variables['water_u'][:].data)
        v=(enc.variables['water_v'][:].data)
        u=np.where(u==-30000,np.nan,u)
        v=np.where(v==-30000,np.nan,v)
        us0.append(u[:,0,:,:]);vs0.append(v[:,0,:,:])
        us2.append(u[:,2,:,:]);vs2.append(v[:,2,:,:])
        us4.append(u[:,4,:,:]);vs4.append(v[:,4,:,:])
        us6.append(u[:,6,:,:]);vs6.append(v[:,6,:,:])
        us8.append(u[:,8,:,:]);vs8.append(v[:,8,:,:])
        us10.append(u[:,10,:,:]);vs10.append(v[:,10,:,:])
        dates.append(dt0)
        del u, v, dt0
        enc.close()
        print(fn)
    dt_hycom=[pd.to_datetime('2000-01-01')+\
              pd.to_timedelta(int(h+9),unit='h') for h in np.concatenate(dates).round(0)]
    dt_hycom=np.array(dt_hycom).flatten()
    u0=np.concatenate(us0);    v0=np.concatenate(vs0)
    u2=np.concatenate(us2);    v2=np.concatenate(vs2)
    u4=np.concatenate(us4);    v4=np.concatenate(vs4)
    u6=np.concatenate(us6);    v6=np.concatenate(vs6)
    u8=np.concatenate(us8);    v8=np.concatenate(vs8)
    u10=np.concatenate(us10);  v10=np.concatenate(vs10)
    return longitude,latitude,dt_hycom,u0,v0,\
        u2, v2,u4, v4,u6, v6, u8, v8,u10, v10



def pred_intp_ecmwf(df,dt_ecmwf,wu,wv,lat1,lon1):
    org_df=df.copy()
    org_df['times']=pd.to_datetime(org_df['times'])
    intp_idx=org_df.loc[org_df[['wu','wv']].isnull().any(1),'times']
    n=len(intp_idx)
    for idx,df_time in enumerate(intp_idx):
        print(idx,'/',n)
        o=org_df.loc[org_df.times==df_time,['Longitude','Latitude']]
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
                temp1=export_date_ecmwf(
                                      pd.to_datetime(df_time),wu,wv,lon1,lat1,dt_ecmwf)            
            #interpolation
            org_df.loc[org_df.times==df_time,'wu']=\
                gstat_idw(
                    temp1['lon'],temp1['lat'],temp1['wu'],o.Longitude,o.Latitude)        
            org_df.loc[org_df.times==df_time,'wv']=\
                gstat_idw(
                    temp1['lon'],temp1['lat'],temp1['wv'],o.Longitude,o.Latitude)
        except:
            org_df.loc[org_df.times==df_time,['wu','wv']]=np.nan
    return org_df

def pred_intp_hycom_v1(df,dt_hycom_v1,u_v1,v_v1,ssh,lat2,lon2):
    org_df=df.copy()
    org_df['times']=pd.to_datetime(org_df['times'])
    if sum([not i in [2016,2017] for i in org_df['times'].dt.year])!=0:
        intp_idx=org_df.loc[org_df[['u','v','ssh']].isnull().any(1),'times']
        n=len(intp_idx)
        for idx,df_time in enumerate(intp_idx):
            print(idx,'/',n)
            o=org_df.loc[org_df.times==df_time,['Longitude','Latitude']]
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
                    temp2=export_date_hycom_v1(
                        pd.to_datetime(df_time), u_v1, v_v1, ssh, lon2, lat2, dt_hycom_v1)
                org_df.loc[org_df.times==df_time,'u_v1']=\
                    gstat_idw(
                        temp2['lon'],temp2['lat'],temp2['u'],o.Longitude,o.Latitude)        
                org_df.loc[org_df.times==df_time,'v_v1']=\
                    gstat_idw(
                        temp2['lon'],temp2['lat'],temp2['v'],o.Longitude,o.Latitude)
                org_df.loc[org_df.times==df_time,'ssh_v1']=\
                    gstat_idw(
                        temp2['lon'],temp2['lat'],temp2['ssh'],o.Longitude,o.Latitude)
            except:
                org_df.loc[org_df.times==df_time,['u_v1','v_v1','ssh_v1']]=np.nan
    else:
        print('2016 17년 자료밖에 없습니다.')