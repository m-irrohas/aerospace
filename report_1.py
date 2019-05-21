import numpy as np ##いつもの
import matplotlib.pyplot as plt ##いつもの
from mpl_toolkits.mplot3d import Axes3D ##三次元プロット用

#Constant
##慣性モーメント
Ix = 1.9#[kgm^2]
Iy = 1.6
Iz = 2.0

#x = [q0,q1,q2,q3,wx,wy,wz]
def get_dot_omega(x):
    ##前状態からの変化量(ω用)
  dot_wx = (Iy-Iz)/Ix*x[5]*x[6]
  dot_wy = (Iz-Ix)/Iy*x[6]*x[4]
  dot_wz = (Ix-Iy)/Iz*x[4]*x[5]
  return (dot_wx,dot_wy,dot_wz)

def get_dot_quotanian(x):
    ##前状態からの変化量(q用)
  A = np.array([[-x[1],-x[2],-x[3]],
  [x[0],-x[3],x[2]],
  [x[3],x[0],-x[1]],
  [-x[2],x[1],x[0]]])
  B = np.array([x[4],x[5],x[6]])
  q_dot = 1/2*np.dot(A,B)
  return (q_dot[0],q_dot[1],q_dot[2],q_dot[3])

def runge_kutta(x0,t0,dt,endt):
  ##初期条件と時刻の状態による4次のルンゲクッタ法
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

def main():
  ##初期条件
  q0_0 = 1.0
  q1_0 = 0.0  
  q2_0 = 0.0
  q3_0 = 0.0
  wx0 = 0.1
  wy0 = 0.1+17.0*2.0*np.pi/60.0
  wz0 = 0.0
  x0 = np.array([q0_0,q1_0,q2_0,q3_0,wx0,wy0,wz0])
  ##その他パラメータ
  t0 = 0.0##初期時間
  dt = 0.1##刻み(粗めでもいい結果でした)
  endt = 100.0##終了時間(これ以上長くするとグラフがつぶれたからこのへんで打ち止め)
  t,q0,q1,q2,q3,wx,wy,wz = runge_kutta(x0,t0,dt,endt)
  
  #Plotting
  #Quaternion
  plt.plot(t,q0,label='q0',color='red')
  plt.plot(t,q1,label='q1',color='indigo')
  plt.plot(t,q2,label='q2',color='pink')
  plt.plot(t,q3,label='q3',color='crimson')
  plt.legend(loc='upper right')
  plt.show()
  #ωx
  plt.plot(t,wx,label='ωx',color='r')
  plt.plot(t,wy,label='ωy',color='b')
  plt.plot(t,wz,label='ωz',color='g')
  plt.legend(loc='upper right')
  plt.show()
  #ωy
  plt.plot(t,wx,label='ωx',color='r')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t,wy,label='ωy',color='b')
  plt.legend(loc='upper right')
  plt.show()
  #ωz
  plt.plot(t,wz,label='ωz',color='g')
  plt.legend(loc='upper right')
  plt.show()
  #3Dplot(w)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.plot(wx,wy,wz,color='black')
  ax.set_xlabel('ωx')
  ax.set_ylabel('ωy')
  ax.set_zlabel('ωz')
  plt.show()

if __name__ == '__main__':
    main() 