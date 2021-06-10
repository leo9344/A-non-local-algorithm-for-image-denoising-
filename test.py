# -*- coding:utf-8 -*-
# @Time   : 2021/6/3 20:06 
# @Author : Leo Li
# @Email  : 846016122@qq.com
import numpy as np
x = np.matrix([-1,-1,0,2,0,-2,0,0,1,1],dtype="float")
x = x.reshape(2,5)
print(x*x.transpose())
s,v,d = np.linalg.svd(0.2*x*x.transpose())
print(s)
print(v)
print(d)