import numpy as np

def A(wx,wy,wz,q0,q1,q2,q3):
  return np.array([[0,-1/2*wx,-1/2*wy,-1/2*wz,-1/2*q1,-1/2*q2,-1/2*q3]
  ,[1/2*wx,0,1/2*wz,-1/2*wy,1/2*q0,-1/2*q3,1/2*q2]
  ,[1/2*wy,-1/2*wz,0,1/2*wx,1/2*q3,1/2*q0,-1/2*q1]
  ,[]])