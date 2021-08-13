import numpy as np
import matplotlib.pyplot as plt  

def plot_result(y1, y2):
    x = [0,1,2,3,4,5,6,7,8,9]
    x = [str(i) for i in x]
    
    plt.plot(x,y1,'ro-',x,y2,'g+-')

    plt.xlabel('class')
    plt.ylabel('accuracy(%)')
    plt.xlim([-1,10])
    plt.ylim([0.8,1])
    plt.legend()
    plt.grid()
    plt.show()
