#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:26:45 2019

@author: tina
"""
from itertools import compress
import numpy as np
def seperation_genre(x,y_train):
    a1 = np.array([1,0,0,0,0,0,0,0,0])
    u1 = ((np.equal(y_train, a1))[:,0]==True)
    a2 = np.array([0,0,0,0,1,0,0,0,0])
    u2 = ((np.equal(y_train, a2))[:,4]==True)
    a3 = np.array([0,0,0,0,0,1,0,0,0])
    u3 = ((np.equal(y_train, a3))[:,5]==True)
    a4 = np.array([0,0,0,0,0,0,1,0,0])
    u4 = ((np.equal(y_train, a4))[:,6]==True)
    label = np.zeros((np.shape(y_train)[0],4))
    m = np.concatenate((u1.reshape(u1.size,1),u2.reshape(u1.size,1),u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,1,1,1])
    m = np.concatenate((u1.reshape(u1.size,1),~u2.reshape(u1.size,1),u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,0,1,1])
    m = np.concatenate((u1.reshape(u1.size,1),u2.reshape(u1.size,1),~u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,1,0,1])
    m = np.concatenate((u1.reshape(u1.size,1),u2.reshape(u1.size,1),u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,1,1,0])
    m = np.concatenate((u1.reshape(u1.size,1),~u2.reshape(u1.size,1),~u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,0,0,1])
    m = np.concatenate((u1.reshape(u1.size,1),~u2.reshape(u1.size,1),u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,0,1,0])
    m = np.concatenate((u1.reshape(u1.size,1),u2.reshape(u1.size,1),~u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,1,0,0])
    m = np.concatenate((u1.reshape(u1.size,1),~u2.reshape(u1.size,1),~u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([1,0,0,0])
    m = np.concatenate((~u1.reshape(u1.size,1),u2.reshape(u1.size,1),u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,1,1,1])
    m = np.concatenate((~u1.reshape(u1.size,1),~u2.reshape(u1.size,1),u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,0,1,1])
    m = np.concatenate((~u1.reshape(u1.size,1),u2.reshape(u1.size,1),~u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,1,0,1])
    m = np.concatenate((~u1.reshape(u1.size,1),u2.reshape(u1.size,1),u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,1,1,0])
    m = np.concatenate((~u1.reshape(u1.size,1),~u2.reshape(u1.size,1),~u3.reshape(u1.size,1),u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,0,0,1])
    m = np.concatenate((~u1.reshape(u1.size,1),~u2.reshape(u1.size,1),u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,0,1,0])
    m = np.concatenate((~u1.reshape(u1.size,1),u2.reshape(u1.size,1),~u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    label[list(compress(range(len(n)), n)),:] = np.array([0,1,0,0])
    m = np.concatenate((~u1.reshape(u1.size,1),~u2.reshape(u1.size,1),~u3.reshape(u1.size,1),~u4.reshape(u1.size,1)), axis=1)
    n = np.all(m,axis=1)
    n = list(compress(range(len(n)), n))
    label = np.delete(label,n,0)
    x_out = np.delete(x,n,0)
    return x_out, label