import numpy as np
import matplotlib.pyplot as plt
import math
def sign(x):
  if x<0:
    return -1
  elif x>0:
    return 1
  else:
    return 0
#constant
a = 1.0
mx = 51
nlast = 30
cfl = 0.5
dx = 0.02
dt = cfl*dx
travel = dt*a*float(nlast)
ecp = 0.01
#dimension
x = np.zeros(mx+1)
yi = np.zeros_like(x)
ye = np.zeros_like(x)
ys1 = np.zeros_like(x)
ys2 = np.zeros_like(x)
ys3 = np.zeros_like(x)
flux = np.zeros_like(x)
work = np.zeros_like(x)
#set grid
#次元に注意
for i in np.arange(0,mx+1):
  x[i] = dx*float(i-1)

for i in np.arange(0,mx+1):
  if x[i]<0.5:
    yi[i] = 1.0

for i in np.arange(0,mx+1):
  if x[i]<0.5+travel:
    ye[i]=1.0

'''scheme by minmod1'''
for i in np.arange(0,mx+1):
  ys1[i] = yi[i]

for j in range(nlast):
  for i in np.arange(1,mx-1):
    dm = ys1[i]-ys1[i-1]
    d0 = ys1[i+1]-ys1[1]
    dp = ys1[i+2]-ys1[i+1]
    s = sign(dm)
    q = s*max(0.0,min(s*dm,s*d0,s*dp))
    aa = abs(a)
    if aa<ecp:
      aa = (a**2+ecp**2)*0.5/ecp
    f = -(dt*a**2/dx*q+aa*(d0-q))
    flux[i] =0.5*(a*ys1[i]+a*ys1[i+1]+f)

  for i in np.arange(2,mx):
    work[i] = ys1[i]-(dt/dx)*(flux[i]-flux[i-1])

  for i in np.arange(2,mx):
    ys1[i] = work[i]

'''solve by scheme by minmod2'''
for i in np.arange(mx+1):
  ys2[i] = yi[i]

for j in range(nlast):
  for i in np.arange(1,mx-1):
    dm = ys2[i]-ys2[i-1]
    d0 = ys2[i+1]-ys2[i]
    dp = ys2[i+2]-ys2[i+1]
    s = sign(dm)
    q = s*max(0.0,min(2.0*s*dm ,2.0*s*d0 ,2.0*s*dp ,0.5*s*(dm+dp))) #0.5いる？？？
    abs_a = abs(a)
    if abs_a < ecp:
      abs_a = (a**2+ecp**2)*0.5/ecp
    f = -(dt*a**2/dx*q + abs_a*(d0-q))
    flux[i] = 0.5*(a*ys2[i]+a*ys2[i+1]+f)

  for i in np.arange(2,mx):
    work[i] = ys2[i]- cfl*(flux[i]-flux[i-1])
  for i in np.arange(2,mx):
    ys2[i] = work[i]

'''solve sceme by superbee'''
for i in np.arange(mx+1):
  ys3[i] = yi[i]

for j in range(nlast):
  for i in np.arange(1,mx-1):
    dm = ys3[i]-ys3[i-1]
    d0 = ys3[i+1]-ys3[i]
    dp = ys3[i+2]-ys3[i+1]
    s = sign(dm)
    sb1 = s*max(0.0,min(2.0*abs(dm),s*d0),min(abs(dm),2*s*d0))
    s = sign(d0)
    sb2 = s*max(0.0,min(2.0*abs(d0),s*dp),min((abs(d0),2.0*s*dp)))
    q = sb1+sb2-d0
    ac = abs(a)
    if ac < ecp:
      ac = (a**2+ecp**2)*0.5/ecp
    f = -(dt*a**2/dx*q + ac*(d0-q))
    flux[i] = 0.5*(a*ys3[3] + a*ys3[i+1] +f)

  for i in np.arange(2,mx):
    work[i] = ys3[i]- cfl*(flux[i]-flux[i-1])
  for i in np.arange(2,mx):
    ys3[i] = work[i]




plt.scatter(x,ys1,marker='o',label='minmod 1',color='r')
plt.scatter(x,ys2,marker='^',label='minmod 2',color='g')
plt.scatter(x,ys3,marker='v',label='superbee',color='b')
plt.plot(x,ye,label='exact')
plt.plot(x,yi,marker='_',label='initial')
plt.title('1D linear-hyperbolic problem:cfl0.5,Step50')
plt.legend()
plt.xlim(0.4,1.0)
plt.ylim(-0.2,1.2)
plt.show()