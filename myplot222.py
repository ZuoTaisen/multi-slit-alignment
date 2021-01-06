import matplotlib.pyplot as plt
import random
#from sys import argv
#fileDir, xlabel, ylable = argv
#%matplotlib inline
def plot(x,y,error_bar = [],label = '1258',xlabel = 'Pixel_number',\
ylabel = 'Statistics',save = False, save_name = 'savedPlot.png',save_format = 'png',labels = []):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    markers = ['o','*','s','|','d','X','p','v','.','1','2','3','4']
    if not hasattr(y[0],'__iter__'): #lines == 1: 
        if hasattr(error_bar,'__iter__'):
            ax.errorbar(x,y,error_bar,capsize = 4,label = label,marker = random.choice(markers))
        else:            
            ax.plot(x,y,label = label,marker = random.choice(markers))
    else:
        if len(error_bar) > 0: #hasattr(error_bar,'__iter__'):
            for i,item in enumerate(y):
                #labels = 'Line' + str(i)
                ax.errorbar(x,item,error_bar[i],capsize = 4,label = labels[i],marker = markers[i]) 
        else:                
            for i,item in enumerate(y):
                #labels = 'Line' + str(i)
                ax.plot(x,item,label = labels[i],marker = markers[i])
    ax.set_xlabel(xlabel,size = 20)
    ax.set_ylabel(ylabel,size = 20)
    spines = ['left','right','top','bottom']
    for ispine in spines:
        plt.gca().spines[ispine].set_linewidth(2)
    plt.rc('font',family = 'Times New Roman')
    plt.tick_params(axis = 'both',direction = 'in',labelsize = 20,width = 2)
    plt.legend(fontsize = 20,frameon = False,loc = 'upper right')
    if save is True:
        plt.savefig(save_name,dpi = 600, format = save_format,bbox_inches = 'tight', transparent = True)
    
