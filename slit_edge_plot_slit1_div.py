# -*- coding: utf-8 -*-
"""
Created on Mon July  11 11:26:23 2020
The program find the peaks and edges of the projection image of the multi-slit and then
 calculate the average positon and inclination angle of the multi-slits in a whole
"""

###################### imports ##############################    
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import peakutils
import math
import os
from scipy.interpolate import interp1d
from PIL import Image
import pyautogui as pg
import time
from imp import reload
from scipy import optimize
import myplot222
###################### imports ############################## 

###################### initial parameters ###################
peak_width = 800
sum_peaks = 1
sum_y_range = 100
ndiv_y = 5
discarded_margin = 600
###################### initial parameters ####################

###################### Functions ##############################

def img_import(img_dir):# = r'D:\work\workRelated\Oliver\Gratingimage109_5deg.png'):
    '''导入图片并返回图片的灰度值矩阵'''
    img = cv2.imread(img_dir)#r'D:\work\workRelated\Oliver\Gratingimage109_5deg.png') #导入图片
    img_gray_matrix = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #获得图片灰度值
    return(img_gray_matrix)
    
def guassian(x,A,sigma,mu):
    return A*np.exp(-(x-mu)**2/sigma**2)   

def guassian_fit(xx,yy):
    '''对一维数据xx，yy进行高斯峰拟合，返回峰强A,峰宽sigma和峰位mu'''
    sigma_guess = (xx[-1]-xx[0])
    mu_guess = np.average(xx)
    aa,bb = optimize.curve_fit(lambda x,A,sigma,mu: guassian(x,A,sigma,mu),xx,yy,p0=[3000,sigma_guess,mu_guess])
    return aa[0],aa[1],aa[2]

def find_peaks(arr,peak_width):
    '''根据输入的峰的宽度peak_width，找到输入的一列数据arr的峰位所在的点，返回峰位点的列表'''
    sum_match = []
    peak_pos = []
    peak_pos2 = []
    for match in np.arange(len(arr)-peak_width):
        upper = match+peak_width
        summ = np.sum(arr[match:upper])
        
        sum_match.append((match,summ))
    for item in sum_match[peak_width:-peak_width]:
        if sum_match[item[0]][1] > sum_match[item[0]-1][1] and sum_match[item[0]][1] > sum_match[item[0]+1][1]:
            up_bound = item[0]+peak_width
            matched_pos = np.array(sum_match[item[0]:up_bound])
            ave_pos = np.average(matched_pos[:,0])
            peak_pos2.append((ave_pos,item[1]))
            #print(sum_match[item[0]:up_bound])
            peak_pos.append((item[0]+int(peak_width/2),item[1]))
            
            #yy = arr[item[0]:up_bound]
            #plt.plot(xx,yy)
            #plt.xlabel('pixel')
            #A,sigma,mu = guassian_fit(xx,yy)
            #plt.plot(xx,guassian(xx,A,sigma,mu))
    return peak_pos

def get_big_peaks(arr,peak_pos,num,peak_width):
    '''在一列数据arr中，已知所有的峰的粗略位置peak_pos[0]和峰强peak_pos[1]，和设定的peak_width，
    利用高斯拟合找到前num个最强峰的精确位置，返回最强峰位置和峰高的二维数组'''
    peak_height_sorted = sorted(peak_pos,key = lambda x: x[1], reverse = True)
    big_peaks_sorted = sorted(peak_height_sorted[:num],key = lambda x: x[0])
    mu_array = []
    #print(big_peaks_sorted,'--------')
    big_peaks_array = np.array(big_peaks_sorted)
    #print(big_peaks_array,'--------')
    for i,peaks in enumerate(big_peaks_array[:,0]):
        low_bound,high_bound = int(peaks-peak_width/2),int(peaks+peak_width/2)
        xx = np.arange(low_bound,high_bound)
        yy = arr[low_bound:high_bound]     
        plt.plot(xx,yy)
        A,sigma,mu = guassian_fit(xx,yy)
        #plt.plot(xx,guassian(xx,A,sigma,mu))
        big_peaks_array[i,0] = mu 
        mu_array.append(mu)
    out3 = np.ones(big_peaks_array.shape)
    out3[:,0] = mu_array
    out3[:,1] = big_peaks_array[:,1]
    return out3

def get_y_ranges(sumed_array_y,ndiv,discarded_margin):
    '''
    将竖直方向的像素点前后去掉discarded_margin后，分成ndiv份, 返回ndiv个ranges的列表
    '''    
    length = len(sumed_array_y)
    maximum = np.max(sumed_array_y)/2
    for left_i in range(length - 1):
        if sumed_array_y[left_i] < maximum and sumed_array_y[left_i+1] >= maximum:
            left_bound = left_i
    for right_i in range(length):
        if sumed_array_y[length - right_i-1] < maximum and sumed_array_y[length - right_i-2] >= maximum:
            right_bound = length - right_i            
    divided_width = int((right_bound - left_bound)/ndiv)
    ranges_out = []
    for i in range(ndiv):
        i_range = left_bound + np.array(np.arange(divided_width)) + i*divided_width
        ranges_out.append(i_range)    
    return ranges_out

def get_div_sum(im_array, y_ranges_array):
    '''
    以y_ranges_array为索引对输入的矩阵进行y方向的分割，对分割后的矩阵对y方向进行积分，然后分别根据peak_width
    找到x方向最强的sum_peaks个峰,返回y_ranges中每个range的y的平均值组成的列表和每个y方向range的
    最强sum_peaks个峰x位置的二维列表
    '''
    global peak_width
    global sum_peaks
    div_yarrays_averaged = []
    div_arrays_ysumed = []
    big_peaks_array = []
    for i,sub_y_array in enumerate(y_ranges_array):
        div_yarrays_averaged.append(np.average(sub_y_array))        
        divided_array = im_array[sub_y_array]
        div_arrays_ysumed.append(np.sum(divided_array,axis = 0))
        #print('y range array =',len(div_arrays_ysumed[i]))
        #plt.plot(np.array(np.arange(4112)),div_arrays_ysumed[i])
        peak_pos = find_peaks(div_arrays_ysumed[i],peak_width)
        big_peaks = get_big_peaks(div_arrays_ysumed[i],peak_pos,sum_peaks,peak_width)
        big_peaks_array.append(big_peaks)
    return div_yarrays_averaged,np.array(big_peaks_array)

def line_fit(x_array,y_array):
    '''
    x_array and y_array 包含对应的x坐标和y坐标的信息，
    对x_array和y_array矩阵的每一列的对应数据进行线性拟合，
    得到斜率k和截距b
    '''
    hori_dim = len(x_array)
    k = np.zeros(hori_dim)
    b = np.zeros(hori_dim)
    for xAxis in range(hori_dim):
        [k[xAxis],b[xAxis]] = np.polyfit(y_array[xAxis],x_array[xAxis],1)
    return k,b

def rotate_angle(k):   
    '''每个峰的半高宽的x坐标（edge_line）和y坐标(partion*partitions)，
    拟合得到图像中条纹的倾斜角度，输出每条边线的倾斜角度'''
    #angle = []
    #for element in range(len(k)):  #利用斜率计算角度
    angle = math.atan(k)*180/3.14159
    return angle
    
def plot_data(image_dir):  
    #print(image_dir)
    ##got the matrix and x_dim y_dim of the image
    gray_matrix = img_import(img_dir = image_dir)    
    im_array = np.array(gray_matrix)
    x_dim,y_dim = len(im_array[0,:]) ,len(im_array[:,0]) 
    #print('the dim is :', x_dim,y_dim)
    ###############################
    
    ##got the average position of the strips in the image
    sumed_array = np.sum(im_array,axis = 0)
    #sumed_all = np.sum(im_array)
    #print('The intensity is {:10.2f}'.format(sumed_all))
    peaks = find_peaks(sumed_array,peak_width)
    peaks = np.array(peaks)
    big_peaks = get_big_peaks(sumed_array,peaks,sum_peaks,peak_width)
    ave_pos = np.average(big_peaks[:,0])
    #################################
    
   
    sumed_array_y = np.sum(im_array,axis = 1)
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(sumed_array)
    #ax1.set_xlim(1500,2500)
    ax2.scatter(big_peaks[:,0],big_peaks[:,1])

    '''
    ###calculate the tilt angle
    y_ranges_array = get_y_ranges(sumed_array_y,ndiv_y,discarded_margin)
    div_y_array,big_peaks_arry = get_div_sum(im_array, y_ranges_array)
    #y_array,big_peaks2 = get_div_sum(im_array, y_ranges)
    
    #fig2 = plt.figure()
    #ax = fig2.add_subplot(111)
    #for i in range(len(div_y_array)):
        #ax.scatter(big_peaks_arry[i][:,0],big_peaks_arry[i][:,1]*i,s = 8)
        #ax.set_xlim(2935,2938)
    plt.savefig('scatter.png',format = 'png',dpi = 300)    
    big_peaks_x = big_peaks_arry[:,:,0].T
    big_peaks_y = np.array([div_y_array]*sum_peaks)
    #plt.plot(big_peaks_x,big_peaks_y)
    k,b = line_fit(big_peaks_x,big_peaks_y)
    angle = rotate_angle(k)
    print('The rotational angle is {:10.2f}\n'.format(np.average(angle))) 

    return ave_pos,angle

def save_file(data_array,file_name):
    import myplot222
    from imp import reload
    reload(myplot222)
    b = plt.hist(data_array,bins = 12)
    print(b)
    pixels = (b[1][1:]+b[1][:-1])/2
    statistics = b[0]
    print(statistics)
    myplot222.plot(pixels,statistics,label = 'Laser measurements with '+ file_name,save = True, xlabel = 'Angle divergence', save_name = file_name+'.png')
    suma = np.sum(statistics)
    ave_pos = 0
    for ii in range(len(pixels)):
        ave_pos+=pixels[ii]*statistics[ii]/suma        
    with open(file_name+'.dat','w+') as file:
        print('{:<8s} {:<8s} {:<8s}'.format('pixels','statistics','average_pos'),file = file)
        for ii in range(len(pixels)):
            print('{:<8.3f} {:<8.3f} {:<8.3f}'.format(pixels[ii],statistics[ii],ave_pos),file = file) 

for iislit in [4]:
    for jj in [1]:
        image_fold = 'D:\\work\\workRelated\\hezhengqiang\\slit' + str(iislit) + '\\' + str(jj) 
        print(image_fold)
        file_name = 'slit' + str(iislit) + '_' + str(jj)
        pos2 = 0
        pos_array = []
        angle_array = []
        for i in list(range(1,100)):# + list(range(14,100)):
        #while True:    
            if i<10:
                infile = file_name + '_000' + str(i)+'.bmp'
            elif i>=10:
                infile = file_name + '_00' + str(i)+'.bmp'
            #print(infile)
            image_dir = image_fold+os.sep+infile
            print(image_dir)
            try:
                ave_pos,angle = plot_data(image_dir)
                if angle > 1 or angle < -1:
                    continue
            except Exception:
                continue
            pos1 = ave_pos
            angle1 = angle
            pos_array.append(pos1)
            angle_array.append(angle1)
        #file_name_pos = file_name + 'pos'
        #save_file(pos_array,file_name_pos)
        file_name_angle = file_name + 'angle'
        save_file(angle_array,file_name_angle)
'''
        import myplot
        from imp import reload
        b = plt.hist(pos_array,bins = 12)
        reload(myplot)
        pixels = (b[1][1:]+b[1][:-1])/2
        statistics = b[0]
        myplot.plot(pixels,statistics,label = 'Laser measurements with '+ file_name,save = True, save_name = file_name+'.png')
        suma = np.sum(statistics)
        ave_pos = 0
        for ii in range(len(pixels)):
            ave_pos+=pixels[ii]*statistics[ii]/suma
            
        with open(file_name+'.dat','w+') as file:
            print('{:<8s} {:<8s} {:<8s}'.format('pixels','statistics','average_pos'),file = file)
            for ii in range(len(pixels)):
                print('{:<8.3f} {:<8.3f} {:<8.3f}'.format(pixels[ii],statistics[ii],ave_pos),file = file)        

        aa = plt.hist(angle_array,bins = 12)
        pixels_a = (aa[1][1:]+aa[1][:-1])/2
        statistics_a = aa[0]
        #myplot.plot(pixels_a,statistics_a,label = 'Laser measurements with '+ file_name,save = True, save_name = file_name +'angle.png')
        suma_a = np.sum(statistics_a)
        ave_ang = 0
        for ii in range(len(pixels_a)):
            ave_ang+=pixels_a[ii]*statistics_a[ii]/suma_a
            
        with open(file_name+'angle.dat','w+') as file_a:
            print('{:<8s} {:<8s} {:<8s}'.format('pixels_a','statistics_a','average_pos_a'),file = file_a)
            for ii in range(len(pixels_a)):
                print('{:<8.3f} {:<8.3f} {:<8.3f}'.format(pixels_a[ii],statistics_a[ii],ave_ang),file = file_a)        
'''


'''        
image_fold = 'D:\\work\\workRelated\\hezhengqiang\\slit11\\1\\'

file_name = 'slit11_1'
pos2 = 0
pos_array = []
for i in range(1,100):
#while True:    
    if i<10:
        infile = file_name + '_000' + str(i)+'.bmp'
    elif i>=10:
        infile = file_name + '_00' + str(i)+'.bmp'
    #print(infile)
    image_dir = image_fold+os.sep+infile
    ave_pos = plot_data(image_dir)
    pos1 = ave_pos
    #if not pos1 == pos2:
    #    pos_array.append(pos1)
    #    pos2 = pos1
    pos_array.append(pos1)
    pos2 = pos1
    #time.sleep(1)

    #screen_shot = 'screenshot1.png'
    #pos = pg.locateOnScreen(screen_shot)
    #x = pos[0]
    #y = pos[1]

    #pg.click(264, 65)
   # pg.PAUSE = 0.251
    #pg.click(641, 460)
    #pg.click(942, 763)
   # pg.PAUSE = 0.237
    #pg.click(1032, 546)


b = plt.hist(pos_array,bins = 12)
reload(myplot)
pixels = (b[1][1:]+b[1][:-1])/2
statistics = b[0]
myplot.plot(pixels,statistics,label = 'Laser measurements with '+ file_name,save = True, save_name = file_name+'.png')
suma = np.sum(statistics)
ave_pos = 0
for ii in range(len(pixels)):
    ave_pos+=pixels[ii]*statistics[ii]/suma
    
with open(file_name+'.dat','w+') as file:
    print('{:<8s} {:<8s} {:<8s}'.format('pixels','statistics','average_pos'),file = file)
    for ii in range(len(pixels)):
        print('{:<8.3f} {:<8.3f} {:<8.3f}'.format(pixels[ii],statistics[ii],ave_pos),file = file)
'''       




