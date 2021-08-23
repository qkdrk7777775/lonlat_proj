# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 21:57:58 2021

@author: cjcho
"""
import numpy as np
df=np.array([[[8,1,7],[4,3,9],[5,2,6]],[[3,1,5],[4,2,9],[1,2,4]]])

def my_fun(a):
    return((a[0]*1/3+a[1]*2/3),(a[0]*2/3+a[1]*1/3))
np.apply_along_axis(np.mean,0,df)


import pandas as pd
pd.Series([3,np.nan,np.nan,8]).interpolate(method='linear')
