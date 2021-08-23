# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:17:27 2021

@author: cjcho
"""
import lonlatProj
import os
import numpy as np
import re

model_dir='D:/numeric'

#load ecmwf
ecmwf_dir=rf'{model_dir}\ERA5'
lon1,lat1,dt_ecmwf,wu,wv=lonlatProj.read_ecmwf(ecmwf_dir=ecmwf_dir)
#np.savez('D:\output/ecmwf',lon1=lon1,lat1=lat1,dt_ecmwf=dt_ecmwf,wu=wu,wv=wv)
#o=np.load('D:\output/ecmwf.npz',allow_pickle=True)
#lon1=o['lon1'];lat1=o['lat1'];dt_ecmwf=o['dt_ecmwf'];wu=o['wu'];wv=o['wv']

#load hycom1
hycom_dir=rf'{model_dir}\HYCOM/ALL'
lon2,lat2,dt_hycom,u,v,ssh=lonlatProj.read_hycom_v1(hycom_dir=hycom_dir)
#np.savez('D:\output/hycom_v1',lon2=lon2,lat2=lat2,dt_hycom=dt_hycom,u=u,v=v,ssh=ssh)
#o=np.load('D:\output/hycom_v1.npz',allow_pickle=True)
#lon2=o['lon2'];lat2=o['lat2'];dt_hycom=o['dt_hycom'];u=o['u'];v=o['v'];ssh=o['ssh']

hycom_dir=rf'{model_dir}\HYCOM/2021'
lon2,lat2,dt_hycom,u,v,ssh=lonlatProj.read_hycom_v1(hycom_dir=hycom_dir)
#np.savez('D:\output/hycom_v1_2021',lon2=lon2,lat2=lat2,dt_hycom=dt_hycom,u=u,v=v,ssh=ssh)
#o=np.load('D:\output/hycom_v1_2021.npz',allow_pickle=True)
#lon2=o['lon2'];lat2=o['lat2'];dt_hycom=o['dt_hycom'];u=o['u'];v=o['v'];ssh=o['ssh']


#load hycom2
hycom_dir=rf'{model_dir}\HYCOM\3hour/'

for depth in [0,2,4,6,8,10]:
    for year in [2016,2017,2019,2020]:
        hycom_files = [i for i in os.listdir(hycom_dir) 
                       if re.compile(f'^{year}.+nc$').findall(i)]
        hycom_files.sort()
        lon3,lat3,dt_hycom,u,v=lonlatProj.read_hycom_v2_1(
                                    hycom_dir,hycom_files,depth)
        np.savez(f'D:\output/hycom_v2_year_{year}_depth_{depth}',
                 lon3=lon3,lat3=lat3,dt_hycom=dt_hycom,u=u,v=v)

#hycom_v2_depth_0 save #shape이 다른데 귀찮아서 뺌
depth=0
o1=np.load(rf'D:\output/hycom_v2_year_2016_depth_{depth}.npz',allow_pickle=True)
o2=np.load(rf'D:\output/hycom_v2_year_2017_depth_{depth}.npz',allow_pickle=True)
o3=np.load(rf'D:\output/hycom_v2_year_2019_depth_{depth}.npz',allow_pickle=True)
o4=np.load(rf'D:\output/hycom_v2_year_2020_depth_{depth}.npz',allow_pickle=True)
lon3=o1['lon3'];lat3=o1['lat3'];
dt_hycom=np.concatenate((
    np.concatenate((o1['dt_hycom'],o2['dt_hycom'])),
    np.concatenate((o3['dt_hycom'],o4['dt_hycom']))
    ))
idx=[i in o1['lat3'] for i in o3['lat3']]

u=np.concatenate((
    np.concatenate((o1['u'],o2['u'])),
    np.concatenate((o3['u'][:,idx,:],o4['u'][:,idx,:]))
    ))
v=np.concatenate((
    np.concatenate((o1['v'],o2['v'])),
    np.concatenate((o3['v'][:,idx,:],o4['v'][:,idx,:]))
    ))

np.savez(f'D:\output/hycom_v2_depth_{depth}',
         lon3=lon3,lat3=lat3,dt_hycom=dt_hycom,u=u,v=v)
#2021년도 추가
hycom_dir=rf'{model_dir}\HYCOM\3hour(2021)/'
for depth in [0,2,4,6,8,10]:
    hycom_files = [i for i in os.listdir(hycom_dir) 
                   if re.compile(f'nc$').findall(i)]
    hycom_files.sort()
    lon3,lat3,dt_hycom,u,v=lonlatProj.read_hycom_v2_1(
                                hycom_dir,hycom_files,depth)
    np.savez(f'D:\output/hycom_v2_year_2021_depth_{depth}',
             lon3=lon3,lat3=lat3,dt_hycom=dt_hycom,u=u,v=v)

###########################################################
for depth in [0,2,4,6,8,10]:
    o1=np.load(rf'D:\output/hycom_v2_year_2016_depth_{depth}.npz',allow_pickle=True)
    o2=np.load(rf'D:\output/hycom_v2_year_2017_depth_{depth}.npz',allow_pickle=True)
    o3=np.load(rf'D:\output/hycom_v2_year_2019_depth_{depth}.npz',allow_pickle=True)
    o4=np.load(rf'D:\output/hycom_v2_year_2020_depth_{depth}.npz',allow_pickle=True)
    lon3=o1['lon3'];lat3=o1['lat3'];
    o1['lat3'].shape,o3['lat3'].shape
    
    dt_hycom=np.concatenate((
        np.concatenate((o1['dt_hycom'],o2['dt_hycom'])),
        np.concatenate((o3['dt_hycom'],o4['dt_hycom']))
        ))
    idx=[i in o1['lat3'] for i in o3['lat3']]
    
    u=np.concatenate((
        np.concatenate((o1['u'],o2['u'])),
        np.concatenate((o3['u'][:,idx,:],o4['u'][:,idx,:]))
        ))
    v=np.concatenate((
        np.concatenate((o1['v'],o2['v'])),
        np.concatenate((o3['v'][:,idx,:],o4['v'][:,idx,:]))
        ))
    
    np.savez(f'D:\output/hycom_v2_depth_{depth}',
             lon3=lon3,lat3=lat3,dt_hycom=dt_hycom,u=u,v=v)
