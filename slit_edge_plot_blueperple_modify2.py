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
from scipy.interpolate import interp1d
from PIL import Image
import time
import glob
import myplot
from imp import reload
reload(myplot)
peak_width = 80
sum_peaks = 8
ndiv_y = 5
tester = []
trunction = 8   #y 轴截断因子，一般取2-10, 数值越大，y轴上升和下降沿截断越多，数值越小，y轴上升沿和下降沿截断越少
###################### Functions ##############################

def img_import(img_dir):# = r'D:\work\workRelated\Oliver\Gratingimage109_5deg.png'):
    '''导入图片并返回图片的灰度值矩阵'''
    img = cv2.imread(img_dir)#r'D:\work\workRelated\Oliver\Gratingimage109_5deg.png') #导入图片
    img_gray_matrix = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #获得图片灰度值
    return(img_gray_matrix)
    
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
    return peak_pos2

def get_big_peaks(peak_pos,num):
    '''在一列数据arr中，已知所有的峰的粗略位置peak_pos[0]和峰强peak_pos[1]，和设定的peak_width，
    找到前num个最强峰的精确位置，返回最强峰位置和峰高的二维数组'''
    out = sorted(peak_pos,key = lambda x: x[1], reverse = True)
    out2 = sorted(out[:num],key = lambda x: x[0])
    return np.array(out2)        

def get_y_ranges(sumed_array_y,ndiv):
    '''
    将竖直方向的像素点前后去掉discarded_margin后，分成ndiv份, 返回ndiv个ranges的列表
    '''    
    global trunction
    length = len(sumed_array_y)
    maximum = np.max(sumed_array_y)/trunction
    for left_i in range(length - 1):
        if sumed_array_y[left_i] < maximum and sumed_array_y[left_i+1] >= maximum:
            left_bound = left_i
    for right_i in range(length):
        if sumed_array_y[length - right_i-1] < maximum and sumed_array_y[length - right_i-2] >= maximum:
            right_bound = length - right_i 
    print('left bond is {:} and right bond is {:}'.format(left_bound,right_bound))
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
        #plt.plot(range(len(div_arrays_ysumed[i])),div_arrays_ysumed[i])
        #print(div_arrays_ysumed[i])
        peak_pos = find_peaks(div_arrays_ysumed[i],peak_width)
        big_peaks = get_big_peaks(peak_pos,sum_peaks)
        big_peaks_array.append(big_peaks)
    return div_yarrays_averaged,np.array(big_peaks_array)

def line_fit(x_array,y_array):
    '''x_array and y_array 包含对应的x坐标和y坐标的信息，对x_array和y_array矩阵的每一列的对应数据进行线性拟合，得到斜率k和截距b'''
    hori_dim = len(x_array)
    k = np.zeros(hori_dim)
    b = np.zeros(hori_dim)
    for xAxis in range(hori_dim):
        [k[xAxis],b[xAxis]] = np.polyfit(y_array[xAxis],x_array[xAxis],1)
    return k,b

def rotate_angle(k):   
    '''每个峰的半高宽的x坐标（edge_line）和y坐标(partion*partitions)，拟合得到图像中条纹的倾斜角度，输出每条边线的倾斜角度'''
    angle = []
    for element in range(len(k)):  #利用斜率计算角度
        angle.append(math.atan(k[element])*180/3.14159)
    return(angle)   
    
def save_to_file(xx,yy,file_name):
    for x,y in zip(xx,yy):
        print('{:<15.8f}{:>15.8f}'.format(x,y),file = file_name)
        
def get_ave_pos(image_dir,peak_width): 
    '''输入图片路径和peak_width，输出峰的位置的矩阵和平均位置'''
    gray_matrix = img_import(img_dir = image_dir)        
    im_array = np.array(gray_matrix)    

    # calculate the positions    
    sumed_array = np.sum(im_array,axis = 0)        
    peaks = find_peaks(sumed_array,peak_width)
    peaks = np.array(peaks)
    big_peaks = get_big_peaks(peaks,sum_peaks)
    peak_poses = big_peaks[:,0]         
    #sumed_y_array = np.sum(im_array,axis = 1)
    ave_pos = np.average(big_peaks[:,0]) 
    print('The average position is {:10.2f}'.format(ave_pos))
    
    # calculate the angles 
    sumed_array_y = np.sum(im_array,axis = 1)
    y_ranges_array = get_y_ranges(sumed_array_y,ndiv_y)
    div_y_array,big_peaks_arry = get_div_sum(im_array, y_ranges_array)
    big_peaks_x = big_peaks_arry[:,:,0].T
    big_peaks_y = np.array([div_y_array]*sum_peaks)
    k,b = line_fit(big_peaks_x,big_peaks_y)
    angle = rotate_angle(k)
    print(angle)
    print('The rotational angle is {:10.2f}\n'.format(np.average(angle))) 
    return peak_poses, ave_pos, angle

def get_image_dir(file_dir,file_name,iislit):
    slit_n = file_name    
    if iislit <=9:
        image_dir = file_dir + '\\'+ slit_n + '\\'+ slit_n +'_000'+ str(iislit)+'.bmp'
    elif iislit >9 and iislit<=99:
        image_dir = file_dir + '\\'+ slit_n + '\\'+ slit_n +'_00'+ str(iislit)+'.bmp'
    elif iislit >99:
        image_dir = file_dir + '\\'+ slit_n + '\\'+ slit_n +'_0'+ str(iislit)+'.bmp'            
    return(image_dir)   
    
def get_rid_of_bad_points(input_array):
    ave_poses = input_array
    ave_poses2 = []
    ave_poses2.append(ave_poses[0])
    for i in range(1,len(ave_poses)):
        if ave_poses[i] > ave_poses[i-1]-5 and ave_poses[i] < ave_poses[i-1]+5:
            ave_poses2.append(ave_poses[i]) 
    return ave_poses2



for i in [1,4]:
    ave_poses = []
    ave_angles = []
    file_name = 'slit' + str(i)#+'-0.8mm' 
    file_dir = 'D:\\work\\workRelated\\hezhengqiang\\blueperple-modify2'
    file_num = len(glob.glob(file_dir + '\\'+ file_name + '\\'+'*.bmp'))
    for iislit in range(1,file_num):
        image_dir = get_image_dir(file_dir,file_name,iislit)
        peak_poses, ave_pos, ave_angle = get_ave_pos(image_dir,peak_width)
        ave_poses.append(ave_pos)
        ave_angles.append(ave_angle)
    xx = np.arange(len(ave_poses))
    with open('ave_pos_blueperple' + file_name + '.dat','w+') as f:
        save_to_file(xx,ave_poses,f)
    with open('ave_angle_blueperple' + file_name + '.dat','w+') as f:
        for eight_angles in ave_angles:
            for one_angle in eight_angles:
                print('{:8.4f}'.format(one_angle),file = f,end = '')
            print('\n',file = f)
    ave_poses2 = get_rid_of_bad_points(ave_poses)
    b = plt.hist(ave_poses2,bins = 16)
    pixels = (b[1][1:]+b[1][:-1])/2
    statistics = b[0]
    #print(statistics)
    myplot.plot(pixels,statistics,label = 'Laser measurements with '+ file_name,save = True, xlabel = 'pixel', save_name = file_name+'.png')

#    print('{:<15.8f}'.format(mu),file = f)        
'''        
    xx = [i for i in range(len(output))]
    A,sigma,mu = guassian_fit(xx,output)
    yy = guassian(xx,A,sigma,mu)
    myplot222.plot(xx,[output,yy],labels = [slit_n,'Gaussian_fit'], xlabel = 'Time (s)',ylabel = \
                'Summed Intensity', save = True, save_name = slit_n + '_scan_laser.png')
    
    
    with open('slit_plot' + str(i) + '.dat','w+') as f:
        save_to_file(xx,output,f)
        print('{:<15.8f}'.format(mu),file = f)
'''        

    

'''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(sumed_array)
    ax2.scatter(big_peaks[:,0],big_peaks[:,1])
'''

'''calculate the tilt angle'''
    
#y_ranges = get_y_ranges(sumed_y_array,sum_y_range)
#y_array,big_peaks2 = get_div_sum(im_array, y_ranges)
'''
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    for i in range(len(y_array)):
        ax.scatter(big_peaks2[i][:,0],big_peaks2[i][:,1],s = 8)
    plt.savefig('scatter.png',format = 'png',dpi = 300)
'''
#big_peaks_x = big_peaks2[:,:,0].T
#big_peaks_y = np.array([y_array]*sum_peaks)
#k,b = line_fit(big_peaks_x,big_peaks_y)
#angle = rotate_angle(k)
#print('The rotational angle is {:10.2f}\n'.format(np.average(angle)))
    
#plot_data()


#while True:
#    plot_data()
#    time.sleep(1)






