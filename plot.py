import numpy as np
import matplotlib.pyplot as plt  

def plot_result(y1, y2, y3):
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x = range(len(names))
    
    plt.plot(x,y1,'ro-',x,y2,'g+-',x,y3,'.-')

    plt.xlabel('class')
    plt.ylabel('accuracy(%)')
    plt.xticks(x, names, rotation=-45)
    plt.xlim([-1,10])
    plt.ylim([0.3,1])
    plt.legend(["Target Model", "VAE_1", "VAE_2"])
    plt.grid()
    plt.show()

def plot_result_accuracy(y1, y2, y3):
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x = range(len(names))
    
    plt.plot(x,y1,'ro-',x,y2,'g+-',x,y3,'.-')

    plt.xlabel('class')
    plt.ylabel('accuracy(%)')
    plt.xticks(x, names, rotation=-45)
    plt.xlim([-1,10])
    plt.ylim([0.3,1])
    plt.legend(["Target Model", "VAE_1", "VAE_2"])
    plt.grid()
    plt.show()

def plot_result_record(y1, y2, y3):
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x = range(len(names))

    plt.plot(x,y1,'ro-',x,y2,'g+-',x,y3,'.-')

    plt.xlabel('class')
    plt.ylabel('accuracy(%)')
    plt.xticks(x, names, rotation=-45)
    plt.xlim([-1,10])
    plt.ylim([0,1])
    plt.legend(["Target Model", "VAE_1", "VAE_2"])
    plt.grid()
    plt.show()