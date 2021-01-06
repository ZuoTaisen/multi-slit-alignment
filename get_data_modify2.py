# -*- coding: utf-8 -*-
#from imp import reload
#
from imp import reload
import myplot222
reload(myplot222)
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import chisquare
#data = pd.read_csv('Angle.dat',sep = '\s+')
#list_data = []
##myplot.plot(data['slit'],[data['s1'],data['s4'],data['s5'],data['s8']])
#myplot.plot(data['slit'],[data['s1'],data['s4'],data['s5'],data['s8']],xlabel = 'Slit Nunmer',ylabel = 'Slit Angle',labels = ['1','4','5','8'],save = True, save_name = 'Slit_angle.png')        
#   myplot222.plot(x,[da[0],da[2],da[4],da[6]],[da[1],da[3],da[5],da[7]],xlabel = 'Slit Nunmer',ylabel = 'Slit Angle',labels = ['1','4','5','8'],save = True, save_name = 'Slit_angle_error.png')          
import numpy as np
import glob
import sys
import statistics
sys.path.append(r'D:\work\workRelated\hezhengqiang\blueperple_modify2')
files = glob.glob('ave_pos_blueperple*.dat')
files = sorted(files,key = lambda x :float(x[22:24]))
markers = ['o','*','s','|','d','X','p','v','.','1','2','3','4']
outdata = []
variation = []
for i,file in enumerate(files[:]):
    pix = pd.read_csv(file,sep = '\s+',usecols = [0,1],names = ['col1','col2'])
    x = pix['col1']
    y = pix['col2']
    ave_y = np.average(y)
    #outdata.append(ave_y)
    y[y < ave_y-10] = ave_y
    y[y > ave_y+10] = ave_y
    outdata.append(np.average(y))
    variation.append(statistics.variance(y))
    plt.plot(x,y,label = 'slit' + str(i+1),lw = 1,marker = random.choice(markers),markersize = 3)
    plt.ylim(2018,2030)
    plt.xlim(-20,250)
    plt.legend(frameon = False)
    plt.rc('font',family = 'Times New Roman')
    plt.tick_params(axis = 'both',direction = 'in',labelsize = 15,width = 2)
    plt.xlabel('Frames',size = 20)
    plt.ylabel('pixels',size = 20)
    plt.savefig('Frame_variation.png',format = 'png',dpi = 600, transparent = True,bbox_inches = 'tight')
#plt.savefig('blue_perple_laser.png',format = 'png',dpi = 600, transparent = True,bbox_inches = 'tight')
slits = [i for i in range(1,13)]
fig = plt.figure()
ax = fig.subplots()
ax.errorbar(slits,outdata,variation,lw = 1,color = 'red', marker = random.choice(markers),markersize = 4)
ax.set_xlabel('Slits',size = 20)
ax.set_ylabel('Average positions',size = 20)
ax.tick_params(axis = 'both',direction = 'in',labelsize = 15,width = 2)
plt.savefig('average_position.png',format = 'png',dpi = 600, transparent = True,bbox_inches = 'tight')
with open('average_position.dat','w') as f:
    for i in range(len(slits)):
        print('{:10.5f} {:10.5f} {:10.5f}'.format(slits[i],outdata[i],variation[i]),file = f)

'''
    max_pix = np.max(pix['pixels'])
    min_pix = np.min(pix['pixels'])
    ave_pix = np.min(pix['average_pos'])
    error_bar = round((max_pix - min_pix)/4,3)
    outdata.append([])
    outdata[i].append([ave_pix,error_bar])
#    with open(file) as f:
#        for i,line in enumerate(f,-1):
#            if i == -1:
#                continue
#            else:
#                sline = line.split()
#                outdata.append([])
#                #outdata[i].append(file)
#                outdata[i].append(sline[2])
                
for i,item in enumerate(outdata):
    if not item == outdata[i-1]:
        for ttt in item:
           print('{:<8.3f}{:>6.3f}'.format(ttt[0],ttt[1]))
print('\n')
'''
# 
#import cv2
#import matplotlib.pyplot as plt
#import numpy as np
#import scipy.signal
#import peakutils
#import math
#import os
#from scipy.interpolate import interp1d
#from PIL import Image
#import pyautogui as pg
#import time
#from imp import reload
#from scipy import optimize
#
#import myplot
#



##plot the summed image in one dimension
#def img_import(img_dir):# = r'D:\work\workRelated\Oliver\Gratingimage109_5deg.png'):
#    
#    img = cv2.imread(img_dir)#r'D:\work\workRelated\Oliver\Gratingimage109_5deg.png') #导入图片
#    img_gray_matrix = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #获得图片灰度值
#    return(img_gray_matrix)
#
#fold = 'D:\\work\\workRelated\\hezhengqiang\\slit8' 
#tt = []
#file_name = 'slit8_8'
#for i in range(1,15):
#    if i<10:
#        file = file_name + '_000' + str(i)+'.bmp'
#    elif i>=10:
#        file = file_name + '_00' + str(i)+'.bmp'
#    #file = 'slit1_'+'4_000'+str(i)+'.bmp'
#    fold_file = os.path.join(fold,'8',file)
#    print(fold_file)
#    gray_matrix = img_import(fold_file)
#    im_array = np.array(gray_matrix)    
#    sumed_array = np.sum(im_array,axis = 0) 
#    plt.plot(sumed_array,linewidth = 0.1)
#    #plt.plot(summa)
#    plt.xlabel('Pixels')
#    plt.ylabel('Intensity')    
#    plt.xlim(1500,2000)
#    plt.savefig('slit1_plot.png',format = 'png',dpi = 300)

