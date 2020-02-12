#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:20:17 2019

@author: tbryan
"""
import os
import pandas as pd
from datetime import datetime

HomeDirectory = os.getcwd()
os.chdir('Data')
os.chdir('IMS')
directory = os.listdir('1st_test')
os.chdir('1st_test')

data = pd.read_table(directory[1],header = None)
data.columns = ['b1x','b1y','b2x','b2y','b3x','b3y','b4x','b4y']


input('Please press enter')

os.chdir(HomeDirectory)


#MOVED TO JUPYTER NOTEBOOK