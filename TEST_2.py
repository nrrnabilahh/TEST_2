#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np

A = np.array([[2, 5, 7], [4, 3, 8], [6, 2, 9]])
print(A)
    
B = np.array([[18, 13, 30]])
print(B)
    
C = numpy.linalg.solve(A, B)
print(C)


# In[18]:


def dydt(y, t):
    return (-2*y*t)

def  rungeKutta(y0, t0, y, h):

    n=int((y-y0)/h)
    t=t0
    
    for i in range(1, n+1):
        k1 = h*dydt(y0, t)
        k2 = h*dydt(y0 + 0.5 * h, t + 0.5*k1)
        k3 = h*dydt(y0 + 0.5 * h, t + 0.5*k2)
        k4 = h*dydt(y0 + 0.5 * h, t + k3)
    
        t = t + (1.0/6.0)*(k1 + 2*k2 + 2*k3 +k4)
    
        y0 = y0 + h
    return t

y0 = 1
t = 2
y = 5
h = 0.2
print('the value of t at y is : ', rungeKutta(y0, y, t, h))


# In[42]:


import numpy as np
import scipy as sc 
import matplotlib.pyplot as plt


def dydt(y, t):
    dydt =  (-2*y*t)
    
    return dydt
print('the value of t at y is : ', dydt(y, t))

start = 0
end = 10
t = np.linspace(start, end, end)

yinitial = np.linspace(y0, y0, end)
y = sc.integrate.odeint(dydt, y0, t)

plt.figure()
plt.plot(t, y, 'r', label='y')
plt.plot(t, yinitial, 'b--', label='t', linewidth=2)
plt.title(f'y(t) vs t')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend()
plt.show()


# In[41]:


import numpy as np
import scipy as sc 
import matplotlib.pyplot as plt


def dydt(y, t):
    dydt =  (-2*y*t)
    
    return dydt
print('the value of t at y is : ', dydt(y, t))



# In[ ]:




