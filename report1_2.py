import numpy as np
import matplotlib.pyplot as plt

def Analytic_Solve(t):
  #input:time
  #output:x(analytic solve)
  alpha = 0.15
  beta = np.sqrt(0.9775)
  return -0.0641*np.cos(2*t)-0.3205*np.sin(2*t)+np.exp(-alpha*t)*(1.0641*np.cos(beta*t)+(0.3603-1.0641*alpha)/beta*np.sin(beta*t))

def drow_Analitic_Solve(T):
  ##刻みを0.0001
  #drow graph
  t_list=np.arange(0,T,0.0001)
  ans = []
  for t in t_list:
    ans.append(Analytic_Solve(t))

  return plt.plot(t_list,ans,label="Analitic Solve")


def Discrete_time_system_Calc(T,dt):
  ##離散系における状態方程式の近似解を出力
  alpha = 0.15
  beta = np.sqrt(0.9775)
  A_11 = np.cos(beta*dt)-alpha/beta*np.sin(beta*dt)
  A_12 = -np.sin(beta*dt)/beta
  A_21 = np.sin(beta*dt)/beta
  A_22 = np.cos(beta*dt)+alpha/beta*np.sin(beta*dt)
  A_dash =np.exp(-alpha*dt) * np.array([[A_11,A_12],[A_21,A_22]])

  A = np.array([[-0.3,-1],[1,0]])
  A_inv = np.linalg.inv(A)
  I = np.identity(2)
  B = np.array([1,0])
  B_dash = np.dot(A_inv,np.dot(A_dash-I,B))


  x = np.zeros((int(T//dt),2))
  x_change=np.zeros(int(T//dt))
  t = np.zeros(int(T//dt))
  ##初期条件(配列は\dot,そのままの順)
  x[0,0]=0
  x[0,1]=1
  for i in range(len(x)-1):
    time = i*dt
    x_next = np.dot(A_dash,x[i])+B_dash*np.sin(2*time)
    x[i+1][0] = x_next[0]
    x[i+1][1] = x_next[1]
    t[i+1] = time

  for j in range(len(x)):
    x_change[j] = x[j][1]
  
  return x_change,t


def main():
  T = 50.##範囲
  dt1 =1.##刻み(粗め)
  x1,t1 = Discrete_time_system_Calc(T,dt1)
  plt.plot(t1,x1,label="dt=1:Approximate Solution")

  drow_Analitic_Solve(T)
  plt.xlim(0,T)
  plt.legend()
  plt.show()

  dt2 = 0.01#刻み(細かめ)
  x2,t2 = Discrete_time_system_Calc(T,dt2)
  plt.plot(t2,x2,label="dt=0.01:Approximate Solution")
  drow_Analitic_Solve(T)
  plt.xlim(0,T)
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()  
  

