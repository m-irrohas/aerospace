import numpy as np ##いつもの
import matplotlib.pyplot as plt ##いつもの
import numpy.random as rd ##ノイズ作成用
import math 
import random
from mpl_toolkits.mplot3d import Axes3D ##三次元プロット用

#Constant
Ix = 1.9#[kgm^2]
Iy = 1.6
Iz = 2.0
##Bは定まる
B = np.array([[0,0,0]
,[0,0,0]
,[0,0,0]
,[0,0,0]
,[1/Ix,0,0]
,[0,1/Iy,0]
,[0,0,Iz]])
#Noize
sigma_v = 0.01
sigma_w = 0.01

def get_dot_omega(x):
    ##前状態からの変化量を得る関数
    ##ただし、ノイズあり
  noise_wx = rd.normal(0.0,0.01)##正規分布に従う
  noise_wy = rd.normal(0.0,0.01)
  noise_wz = rd.normal(0.0,0.01)
  dot_wx = (Iy-Iz)/Ix*x[5]*x[6]+noise_wx
  dot_wy = (Iz-Ix)/Iy*x[6]*x[4]+noise_wy
  dot_wz = (Ix-Iy)/Iz*x[4]*x[5]+noise_wz
  return (dot_wx,dot_wy,dot_wz)

def get_dot_quotanian(x):
    ##qの変化量を得る関数
  A = np.array([[-x[1],-x[2],-x[3]],
  [x[0],-x[3],x[2]],
  [x[3],x[0],-x[1]],
  [-x[2],x[1],x[0]]])
  B = np.array([x[4],x[5],x[6]])
  q_dot = 1/2*np.dot(A,B)
  return (q_dot[0],q_dot[1],q_dot[2],q_dot[3])

def runge_kutta(x0,t0,dt,endt):
  ##初期条件と時刻の状態による4次のルンゲクッタ法
  ##課題1と同じ
  x = x0
  t = t0
  res_q0 = [x0[0]]
  res_q1 = [x0[1]]
  res_q2 = [x0[2]]
  res_q3 = [x0[3]]
  res_wx = [x0[4]]
  res_wy = [x0[5]]
  res_wz = [x0[6]]
  res_t = [t0]
  while t <= endt:
    k1_q0 = dt*get_dot_quotanian(x)[0]
    k1_q1 = dt*get_dot_quotanian(x)[1]
    k1_q2 = dt*get_dot_quotanian(x)[2]
    k1_q3 = dt*get_dot_quotanian(x)[3]
    k1_wx = dt*get_dot_omega(x)[0]
    k1_wy = dt*get_dot_omega(x)[1]
    k1_wz = dt*get_dot_omega(x)[2]
    x_k1 = x + np.array([k1_q0/2.0,k1_q1/2.0,k1_q2/2.0,k1_q3/2.0,k1_wx/2.0,k1_wy/2.0,k1_wz/2.0])

    k2_q0 = dt*get_dot_quotanian(x_k1)[0]
    k2_q1 = dt*get_dot_quotanian(x_k1)[1]
    k2_q2 = dt*get_dot_quotanian(x_k1)[2]
    k2_q3 = dt*get_dot_quotanian(x_k1)[3]
    k2_wx = dt*get_dot_omega(x_k1)[0]
    k2_wy = dt*get_dot_omega(x_k1)[1]
    k2_wz = dt*get_dot_omega(x_k1)[2]
    x_k2 = x + np.array([k2_q0/2.0,k2_q1/2.0,k2_q2/2.0,k2_q3/2.0,k2_wx/2.0,k2_wy/2.0,k2_wz/2.0])

    k3_q0 = dt*get_dot_quotanian(x_k2)[0]
    k3_q1 = dt*get_dot_quotanian(x_k2)[1]
    k3_q2 = dt*get_dot_quotanian(x_k2)[2]
    k3_q3 = dt*get_dot_quotanian(x_k2)[3]
    k3_wx = dt*get_dot_omega(x_k2)[0]
    k3_wy = dt*get_dot_omega(x_k2)[1]
    k3_wz = dt*get_dot_omega(x_k2)[2]
    x_k3 = x + np.array([k3_q0,k3_q1,k3_q2,k3_q3,k3_wx,k3_wy,k3_wz])

    k4_q0 = dt*get_dot_quotanian(x_k3)[0]
    k4_q1 = dt*get_dot_quotanian(x_k3)[1]
    k4_q2 = dt*get_dot_quotanian(x_k3)[2]
    k4_q3 = dt*get_dot_quotanian(x_k3)[3]
    k4_wx = dt*get_dot_omega(x_k3)[0]
    k4_wy = dt*get_dot_omega(x_k3)[1]
    k4_wz = dt*get_dot_omega(x_k3)[2]

    k_q0 = (k1_q0+k2_q0*2.0+k3_q0*2.0+k4_q0)/6.0
    k_q1 = (k1_q1+k2_q1*2.0+k3_q1*2.0+k4_q1)/6.0
    k_q2 = (k1_q2+k2_q2*2.0+k3_q2*2.0+k4_q2)/6.0
    k_q3 = (k1_q3+k2_q3*2.0+k3_q3*2.0+k4_q3)/6.0
    k_wx = (k1_wx+2.0*k2_wx+2.0*k3_wx+k4_wx)/6.0
    k_wy = (k1_wy+2.0*k2_wy+2.0*k3_wy+k4_wy)/6.0
    k_wz = (k1_wz+2.0*k2_wz+2.0*k3_wz+k4_wz)/6.0

    t += dt
    x = x+np.array([k_q0,k_q1,k_q2,k_q3,k_wx,k_wy,k_wz])
    res_t.append(t)
    res_q0.append(x[0])
    res_q1.append(x[1])
    res_q2.append(x[2])
    res_q3.append(x[3])
    res_wx.append(x[4])
    res_wy.append(x[5])
    res_wz.append(x[6])

  return res_t,res_q0,res_q1,res_q2,res_q3,res_wx,res_wy,res_wz
#X = [q0,q1,q2,q3,wx,wy,wz]
#x = ΔX = [Δq,Δω]

def runge_kutta_section(dt,X):
  ##状態量x_{k-1}(と微小区間)が与えられれば、そのときの出力(x_{k})を返すルンゲクッタ法
  ##離散用にもう一つ作りました。わかりにくくてすいません。
  ##微少量が出力されることに注意
  k1_q0 = dt*get_dot_quotanian(X)[0]
  k1_q1 = dt*get_dot_quotanian(X)[1]
  k1_q2 = dt*get_dot_quotanian(X)[2]
  k1_q3 = dt*get_dot_quotanian(X)[3]
  k1_wx = dt*get_dot_omega(X)[0]
  k1_wy = dt*get_dot_omega(X)[1]
  k1_wz = dt*get_dot_omega(X)[2]
  X_k1 = X + np.array([k1_q0/2.0,k1_q1/2.0,k1_q2/2.0,k1_q3/2.0,k1_wx/2.0,k1_wy/2.0,k1_wz/2.0])

  k2_q0 = dt*get_dot_quotanian(X_k1)[0]
  k2_q1 = dt*get_dot_quotanian(X_k1)[1]
  k2_q2 = dt*get_dot_quotanian(X_k1)[2]
  k2_q3 = dt*get_dot_quotanian(X_k1)[3]
  k2_wx = dt*get_dot_omega(X_k1)[0]
  k2_wy = dt*get_dot_omega(X_k1)[1]
  k2_wz = dt*get_dot_omega(X_k1)[2]
  X_k2 = X + np.array([k2_q0/2.0,k2_q1/2.0,k2_q2/2.0,k2_q3/2.0,k2_wx/2.0,k2_wy/2.0,k2_wz/2.0])

  k3_q0 = dt*get_dot_quotanian(X_k2)[0]
  k3_q1 = dt*get_dot_quotanian(X_k2)[1]
  k3_q2 = dt*get_dot_quotanian(X_k2)[2]
  k3_q3 = dt*get_dot_quotanian(X_k2)[3]
  k3_wx = dt*get_dot_omega(X_k2)[0]
  k3_wy = dt*get_dot_omega(X_k2)[1]
  k3_wz = dt*get_dot_omega(X_k2)[2]
  X_k3 = X + np.array([k3_q0,k3_q1,k3_q2,k3_q3,k3_wx,k3_wy,k3_wz])

  k4_q0 = dt*get_dot_quotanian(X_k3)[0]
  k4_q1 = dt*get_dot_quotanian(X_k3)[1]
  k4_q2 = dt*get_dot_quotanian(X_k3)[2]
  k4_q3 = dt*get_dot_quotanian(X_k3)[3]
  k4_wx = dt*get_dot_omega(X_k3)[0]
  k4_wy = dt*get_dot_omega(X_k3)[1]
  k4_wz = dt*get_dot_omega(X_k3)[2]

  k_q0 = (k1_q0+k2_q0*2.0+k3_q0*2.0+k4_q0)/6.0
  k_q1 = (k1_q1+k2_q1*2.0+k3_q1*2.0+k4_q1)/6.0
  k_q2 = (k1_q2+k2_q2*2.0+k3_q2*2.0+k4_q2)/6.0
  k_q3 = (k1_q3+k2_q3*2.0+k3_q3*2.0+k4_q3)/6.0
  k_wx = (k1_wx+2.0*k2_wx+2.0*k3_wx+k4_wx)/6.0
  k_wy = (k1_wy+2.0*k2_wy+2.0*k3_wy+k4_wy)/6.0
  k_wz = (k1_wz+2.0*k2_wz+2.0*k3_wz+k4_wz)/6.0

  x1 = X+np.array([k_q0,k_q1,k_q2,k_q3,k_wx,k_wy,k_wz])
  return x1[0],x1[1],x1[2],x1[3],x1[4],x1[5],x1[6]

def A(X):
  ##Xを与えるとAを返す
  ##ここでXは絶対量であることに注意
  ##q0=x[0],,,,
  ##wx=x[4],wy=x[5],wz=x[6]
  return np.array([[0,-1/2*X[4],-1/2*X[5],-1/2*X[6],-1/2*X[1],-1/2*X[2],-1/2*X[3]]
  ,[1/2*X[4],0,1/2*X[6],-1/2*X[5],1/2*X[0],-1/2*X[3],1/2*X[2]]
  ,[1/2*X[5],-1/2*X[6],0,1/2*X[4],1/2*X[3],1/2*X[0],-1/2*X[1]]
  ,[1/2*X[6],1/2*X[5],-1/2*X[4],0,-1/2*X[2],1/2*X[1],1/2*X[0]]
  ,[0,0,0,0,0,(Iy-Iz)/Ix*X[6],(Iy-Iz)/Ix*X[5]]
  ,[0,0,0,0,(Iz-Ix)/Iy*X[6],0,(Iz-Ix)/Iy*X[4]]
  ,[0,0,0,0,(Ix-Iy)/Iz*X[5],(Ix-Iy)/Iz*X[4],0]],dtype=float)

def observate_equation(X):
  ##Hは状態量Xに依存
  ##x,Xを与えるとyとHを返す
  ##vを導入
  ##Hのランダム化
  ##どのHが得られたかもほしい(Debug用)
  H = np.zeros((3,7))
  choose_H = random.randint(1,3)
  if choose_H == 1:
    H[0,0] = 2*X[0]
    H[0,1] = 2*X[1]
    H[0,2] = -2*X[2]
    H[0,3] = -2*X[3]
    H[1,0] = 2*X[3]
    H[1,1] = 2*X[2]
    H[1,2] = 2*X[1]
    H[1,3] = 2*X[0]
    H[2,0] = -2*X[2]
    H[2,1] = 2*X[3]
    H[2,2] = -2*X[0]
    H[2,3] = 2*X[1]
  elif choose_H == 2:
    H[0,0] = -2*X[3]
    H[0,1] = 2*X[2]
    H[0,2] = 2*X[1]
    H[0,3] = -2*X[0]
    H[1,0] = 2*X[0]
    H[1,1] = -2*X[1]
    H[1,2] = 2*X[2]
    H[1,3] = -2*X[3]
    H[2,0] = 2*X[1]
    H[2,1] = 2*X[0]
    H[2,2] = 2*X[3]
    H[2,3] = 2*X[2]
  else:
    H[0,0] = 2*X[2]
    H[0,1] = 2*X[3]
    H[0,2] = 2*X[0]
    H[0,3] = 2*X[1]
    H[1,0] = -2*X[1]
    H[1,1] = -2*X[0]
    H[1,2] = 2*X[3]
    H[1,3] = 2*X[2]
    H[2,0] = 2*X[0]
    H[2,1] = -2*X[1]
    H[2,2] = -2*X[2]
    H[2,3] = 2*X[3]
  y = np.dot(H,X)
  noise_v = np.array([rd.normal(0.0,0.01),rd.normal(0.0,0.01),rd.normal(0.0,0.01)])
  y = y+noise_v
  return H,y,choose_H

def Phi(A,dt):
    ##行列指数関数Φ
    ##Scipyを持っていなかったために定義
  err = 1
  exp = np.eye(7)
  S = exp
  Adt = dt*A
  k = 1
  while k < 25:
    exp = 1/k*np.dot(exp,Adt)
    S = S + exp
    err = np.linalg.norm(S)
    k += 1
  return S

def Gamma(A,dt):
    ##Γ
  return np.dot(np.dot(np.linalg.inv(A),Phi(A,dt)-np.eye(7)),B)

def Predict_step(phi,gamma,x_before,P_before):
    ##予測ステップ
    ##入力:Phi(t),Gamma(t),hat{x}_k-1,P_k-1
    ##出力:\hat{x}_k|k-1,P_k|k-1(np.array)
  x_before = np.array(x_before)
  Q = np.array([[sigma_w**2,0,0],[0,sigma_w**2,0],[0,0,sigma_w**2]])#定数といえば定数
  predict_x = np.dot(phi,x_before)
  predict_P = np.dot(np.dot(phi,P_before),phi.T) + np.dot(np.dot(gamma,Q),gamma.T)
  return predict_x , predict_P

def Renewal_step(y,predict_x,H,predict_P):
  ##y:観測値(観測方程式より得られるy)
  ##predict_x:\hat_k|k-1
  ##H:DCM(観測方程式に組み込んだ)
  ##predict_P:P_{k|k-1}
  ##出力:\hat_{x}_k,P_k
  R = np.array([[sigma_v**2,0,0],[0,sigma_v**2,0],[0,0,sigma_v**2]])
  z_k = y - np.dot(H,predict_x)
  S_k = R + np.dot(np.dot(H,predict_P),H.T)
  K_k = np.dot(np.dot(predict_P,H.T),np.linalg.inv(S_k))
  x_k = np.dot(K_k,z_k)
  P_k = np.dot(np.eye(7)-np.dot(K_k,H),predict_P)
  return x_k,P_k


    
def main():
  #initial state
  omega_est_0 = np.array([0.1,0.1+17*2*math.pi/60,0.1])+np.array([0.1,0.1,0.1])
  a = 0.012171
  rest = math.sqrt(a/3)
  q_est_0 = np.array([1-a,rest,rest,rest])
  X_est_0 = np.array([q_est_0[0],q_est_0[1],q_est_0[2],q_est_0[3],omega_est_0[0],omega_est_0[1],omega_est_0[2]])
  X0 = np.array([1,0,0,0,0.1,0.1+17*2*math.pi/60.0,0.1])
  P_0 = np.array([[sigma_v,0,0,0,0,0,0]
  ,[0,sigma_v,0,0,0,0,0]
  ,[0,0,sigma_v,0,0,0,0]
  ,[0,0,0,sigma_v,0,0,0]
  ,[0,0,0,0,sigma_w,0,0]
  ,[0,0,0,0,0,sigma_w,0]
  ,[0,0,0,0,0,0,sigma_w]])
  ##状態推移パラメータ
  t0=0.0
  dt = 0.01
  endt = 50.0
  ##真値系
  t,true_q0,true_q1,true_q2,true_q3,true_wx,true_wy,true_wz = runge_kutta(X0,t0,dt,endt)
  ##推定量の設定
  #q0
  est_q0 = np.zeros_like(true_q0)
  est_q0[0] = X_est_0[0]
  #q1
  est_q1= np.zeros_like(true_q1)
  est_q1[0] = X_est_0[1]
  #q2
  est_q2= np.zeros_like(true_q2)
  est_q2[0] = X_est_0[2]
  #q3
  est_q3= np.zeros_like(true_q3)
  est_q3[0] = X_est_0[3]
  #wx
  est_wx = np.zeros_like(true_wx)
  est_wx[0] = X_est_0[4]
  #wy
  est_wy = np.zeros_like(true_wy)
  est_wy[0] = X_est_0[5]
  #wz
  est_wz = np.zeros_like(true_wz)
  est_wz[0] = X_est_0[6]
  #P
  est_P = P_0
  ##Plot用のPとδ
  est_P_00 = np.zeros_like(t)
  est_P_00[0] = P_0[0,0]
  delta_q0 = np.zeros_like(t)
  est_P_11 = np.zeros_like(t)
  est_P_11[0] = P_0[1,1]
  delta_q1 = np.zeros_like(t)
  est_P_22 = np.zeros_like(t)
  est_P_22[0] = P_0[2,2]
  delta_q2 = np.zeros_like(t)
  est_P_33 = np.zeros_like(t)
  est_P_33[0] = P_0[3,3]
  delta_q3 = np.zeros_like(t)
  est_P_44 = np.zeros_like(t)
  est_P_44[0] = P_0[4,4]
  delta_wx = np.zeros_like(t)
  est_P_55 = np.zeros_like(t)
  est_P_55[0] = P_0[5,5]
  delta_wy = np.zeros_like(t)
  est_P_66 = np.zeros_like(t)
  est_P_66[0] = P_0[6,6]
  delta_wz = np.zeros_like(t)
  est_X = [X_est_0]
  ##以下推定系の計算
  step = len(t)
  choose_H_number = []
  ##t % 1==0となるごとに推定系を更新したい。
  for k in range(step-1):
    est_q0[k+1],est_q1[k+1],est_q2[k+1],est_q3[k+1],est_wx[k+1],est_wy[k+1],est_wz[k+1] = runge_kutta_section(dt,est_X[k])
    norm = math.sqrt(est_q0[k+1]**2+est_q1[k+1]**2+est_q2[k+1]**2+est_q3[k+1]**2)
    est_q0[k+1] = est_q0[k+1]/norm
    est_q1[k+1] = est_q1[k+1]/norm
    est_q2[k+1] = est_q2[k+1]/norm
    est_q3[k+1] = est_q3[k+1]/norm
    est_X.append([est_q0[k+1],est_q1[k+1],est_q2[k+1],est_q3[k+1],est_wx[k+1],est_wy[k+1],est_wz[k+1]])
    ##状態量により得られる各行列
    A_k = A(est_X[k+1])
    Phi_k = Phi(A_k,dt)
    Gamma_k = Gamma(A_k,dt)
    Q = np.array([[sigma_w**2,0,0],[0,sigma_w**2,0],[0,0,sigma_w**2]])
    est_P = np.dot(np.dot(Phi_k,est_P),Phi_k.T)+np.dot(np.dot(Gamma_k,Q),Gamma_k.T)
    ##観測値による予測→更新カルマンフィルタ
    if (k+1)%int(1.0/dt)==0:
      true_X = [true_q0[k+1],true_q1[k+1],true_q2[k+1],true_q3[k+1],true_wx[k+1],true_wy[k+1],true_wz[k+1]]
      H_k,y_k,i = observate_equation(true_X)
      predict_x_k,predict_P_k = Predict_step(Phi_k,Gamma_k,est_X[k+1],est_P)
      est_x_k,P_k = Renewal_step(y_k,predict_x_k,H_k,predict_P_k)
      est_q0[k+1] += est_x_k[0]
      est_q1[k+1] += est_x_k[1]
      est_q2[k+1] += est_x_k[2]
      est_q3[k+1] += est_x_k[3]
      est_wx[k+1] += est_x_k[4]
      est_wy[k+1] += est_x_k[5]
      est_wz[k+1] += est_x_k[6]
      norm = math.sqrt(est_q0[k+1]**2+est_q1[k+1]**2+est_q2[k+1]**2+est_q3[k+1]**2)
      est_q0[k+1] = est_q0[k+1]/norm
      est_q1[k+1] = est_q1[k+1]/norm
      est_q2[k+1] = est_q2[k+1]/norm
      est_q3[k+1] = est_q3[k+1]/norm
      est_X[k+1] = [est_q0[k+1],est_q1[k+1],est_q2[k+1],est_q3[k+1],est_wx[k+1],est_wy[k+1],est_wz[k+1]]
      est_P = P_k
      choose_H_number.append(i)
    est_P_00[k+1] = math.sqrt(est_P[0,0])
    delta_q0[k+1] = true_q0[k+1]-est_q0[k+1]
    est_P_11[k+1] = math.sqrt(est_P[1,1])
    delta_q1[k+1] = true_q1[k+1]-est_q1[k+1]
    est_P_22[k+1] = math.sqrt(est_P[2,2])
    delta_q2[k+1] = true_q2[k+1]-est_q2[k+1]
    est_P_33[k+1] = math.sqrt(est_P[3,3])
    delta_q3[k+1] = true_q3[k+1]-est_q3[k+1]
    est_P_44[k+1] = math.sqrt(est_P[4,4])
    delta_wx[k+1] = true_wx[k+1]-est_wx[k+1]
    est_P_55[k+1] = math.sqrt(est_P[5,5])
    delta_wy[k+1] = true_wy[k+1]-est_wy[k+1]
    est_P_66[k+1] = math.sqrt(est_P[6,6])
    delta_wz[k+1] = true_wz[k+1]-est_wz[k+1]
  ##以下プロット
  plt.plot(t,true_q0,label='truth-q0',color='pink')
  plt.plot(t,est_q0,label='estimated-q0',color='purple')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,true_q1,label='truth-q1',color='pink')
  plt.plot(t,est_q1,label='estimated-q1',color='purple')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,true_q2,label='truth-q2',color='pink')
  plt.plot(t,est_q2,label='estimated-q2',color='purple')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,true_q3,label='truth-q3',color='pink')
  plt.plot(t,est_q3,label='estimated-q3',color='purple')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,true_wx,label='truth-wx',color='pink')
  plt.plot(t,est_wx,label='estimated-wx',color='red')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,true_wy,label='truth-wy',color='pink')
  plt.plot(t,est_wy,label='estimated-wy',color='red')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,true_wz,label='truth-wy',color='pink')
  plt.plot(t,est_wz,label='estimated-wz',color='red')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,est_P_00,label='±sqrt(P_00)',color='lightgreen')
  plt.plot(t,-est_P_00,color='lightgreen')
  plt.plot(t,delta_q0,label='δq0',color='darkgreen')
  plt.legend(loc='upper right')
  plt.legend()
  plt.show()
  plt.plot(t,est_P_11,label='±sqrt(P_11)',color='lightgreen')
  plt.plot(t,-est_P_11,color='lightgreen')
  plt.plot(t,delta_q1,label='δq1',color='darkgreen')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,est_P_22,label='±sqrt(P_22)',color='lightgreen')
  plt.plot(t,-est_P_22,color='lightgreen')
  plt.plot(t,delta_q2,label='δq2',color='darkgreen')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,est_P_33,label='±sqrt(P_33)',color='lightgreen')
  plt.plot(t,-est_P_33,color='lightgreen')
  plt.plot(t,delta_q3,label='δq3',color='darkgreen')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,est_P_44,label='±sqrt(P_44)',color='skyblue')
  plt.plot(t,-est_P_44,color='skyblue')
  plt.plot(t,delta_wx,label='δωx',color='navy')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,est_P_55,label='±sqrt(P_55)',color='skyblue')
  plt.plot(t,-est_P_55,color='skyblue')
  plt.plot(t,delta_wy,label='δωy',color='navy')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,est_P_66,label='±sqrt(P_66)',color='skyblue')
  plt.plot(t,-est_P_66,color='skyblue')
  plt.plot(t,delta_wz,label='δωz',color='navy')
  plt.legend(loc='upper right')
  plt.show()
  print(choose_H_number)

if __name__ == '__main__': 
    main()