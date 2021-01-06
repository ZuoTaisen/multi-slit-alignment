import matplotlib.pyplot as plt
import random
#from sys import argv
#fileDir, xlabel, ylable = argv
#%matplotlib inline
def plot(x,y,label = 'Line 1',xlabel = 'Pixel_number',\
ylabel = 'Statistics',save = False, save_name = 'savedPlot.png',save_format = 'png',labels = []):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    markers = ['o','*','s','|','d','X','p','v','.','1','2','3','4']
    if not hasattr(y[0],'__iter__'): #lines == 1:        
        ax.plot(x,y,label = label,marker = random.choice(markers))
    else:
        for i,item in enumerate(y):
            labels = label + str(i)
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
    
