
import numpy as np
import matplotlib.pyplot as plt  
x=[0,1,2,3,4,5,6,7,8,9]
x = [str(i) for i in x]
y1=[99.08, 99.30, 99.42, 99.31, 96.03, 95.96, 98.43, 96.40, 98.87, 98.81]
y2=[96.53, 99.56, 75.58, 97.62, 86.86, 63.23, 73.28, 71.89, 91.58, 72.45]

l1=plt.plot(x,y1,'r--',label='Target Model')
l2=plt.plot(x,y2,'g--',label='VAEGAN_G')
plt.plot(x,y1,'ro-',x,y2,'g+-')

plt.xlabel('class')
plt.ylabel('accuracy(%)')
plt.xlim([-1,10])
plt.legend()
plt.grid()
plt.show()
