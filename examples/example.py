#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:55:44 2021

@author: maugeais
"""

import burgers
import matplotlib.pyplot as plt
import numpy as np
import time


b=burgers.cburgers('file', T = 0.155, dt = 2e-4, alpha = 2*1.23e-8, mu = 1/8e-3*1.668e-5, nu = 4.65e-13, bore='perce.dat',
                   nComp = -1, customInit ='crescendo.wav', w_len = 1024*16, w_step = 256)

b.out_opt(density = False, movie = False, save = 'crescendo-res.wav')

Lambda0 = b.initial()

t0 = time.time()
Lambda = b.parFEM(Lambda0)
  
print(time.time()-t0)

b=burgers.cburgers('file', T = 0.155, dt = 2e-4, alpha = 2*1.23e-8, mu = 1/8e-3*1.668e-5, nu = 4.65e-13, bore='perce.dat',
                   nComp = -1, customInit ='crescendoDecresendo.wav', w_len = 1024*8, w_step = 256)

b.out_opt(density = False, movie = False, save = 'crescendoDecresendo-res.wav')

Lambda0 = b.initial()

t0 = time.time()
Lambda = b.parFEM(Lambda0)
  
print(time.time()-t0)

