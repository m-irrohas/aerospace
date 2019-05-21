# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 01:36:18 2018

@author: masaya
"""
##なぜかSpyderでしか動かなかった。
##randomモジュールのせい?????
import numpy as np
import math
import numpy.random as rand
import numpy.linalg as lg
import matplotlib.pyplot as plt
#constant
R = 100.0##半径
r = 10##内半径
t = np.arange(81)##Step数=時刻
omega = 2*10*math.pi/(len(t)-1)##t=80で一周するように調整
##ノイズ(講義プリントを参考)
sigma_w = 1.0##小さめ
sigma_r = 5.0##小さめ
sigma_b = math.pi/180##小さめ

R_mat = np.array([[sigma_r**2,0],[0,sigma_b**2]])#R
Q = sigma_w**2*np.eye(2)#Q
A = np.array([[1,0],[0,1]])#今回はConstだから楽
def get_next_x(i,x):
    ##時刻と前の状態量から、次の真値を求める関数。
    ##外乱あり
    ##前の状態から次の状態を得る関数、これが真値系を構成する
    a = r/R*(R-r)*omega
    omega1 = r/R*omega
    omega2 = (1-r/R)*omega
    u = a*np.array([-math.sin(omega1*i)-math.sin(omega2*i),math.cos(omega1*i)-math.cos(omega2*i)])
    x_next = np.dot(A,x) + u +rand.normal(0.0,sigma_w,2)
    return x_next,u

def observate_equation(x):
    ##真値から得られる観測方程式
    ##ノイズあり
    h_pure = h(x)
    y = h_pure + np.array([rand.normal(0.0,sigma_r),rand.normal(0.0,sigma_b)])
    return y

def Ck(x):
    ##hのヤコビアン
    x1,x2 = x[0],x[1]
    Jacobian_Ck = np.zeros((2,2))
    Jacobian_Ck[0,0] = x1/math.sqrt(x1**2+x2**2)
    Jacobian_Ck[0,1] = x2/math.sqrt(x1**2+x2**2)
    Jacobian_Ck[1,0] = -x2/(x1**2+x2**2)
    Jacobian_Ck[1,1] = x1/(x1**2+x2**2)
    return Jacobian_Ck

def h(x):
    ##ただのh
    return np.array([math.sqrt(x[0]**2+x[1]**2),math.atan2(x[1],x[0])])

def EKF_One_Step(mt,Vt,ut,yt_plus_1):
    ##前状態の諸パラメータから、カルマンフィルタにより推定値を更新
    ##yは観測値
    ##他はいつもと同じ
    ##講義プリントをもとに作成
    I = np.eye(len(Vt))
    ##予測ステップ
    m_prime = np.dot(A,mt)+ut
    V_prime = np.dot(A,np.dot(Vt,A.T))+Q
    ##更新ステップ
    C = Ck(m_prime)
    K = np.dot(np.dot(V_prime,C.T),lg.inv(np.dot(np.dot(C,V_prime),C.T)+R_mat))
    m_next = m_prime + np.dot(K , yt_plus_1 - h(m_prime))
    V_next = np.dot(I-np.dot(K,C),V_prime)
    return m_next,V_next

def main1():
    #inicial State
    x0 = np.array([100,0])
    m0 = x0
    V0 = np.zeros((2,2))
    y0 = x0
    
    x = x0
    m = m0
    V = V0
    x1_true = [x0[0]]
    x2_true = [x0[1]]
    x1 = [x0[0]]
    x2 = [x0[1]]
    y1 = [y0[0]]
    y2 = [y0[1]]
    #誤差値
    err_observe = 0
    err_EKF = 0
    
    for i in t-1:
        x_next,ui = get_next_x(i+1,x)
        x1_true.append(x_next[0])
        x2_true.append(x_next[1])
        y_next = observate_equation(x_next)
        m_next,V_next = EKF_One_Step(m,V,ui,y_next)
        x1.append(m_next[0])
        x2.append(m_next[1])
        y1.append(y_next[0]*math.cos(y_next[1]))
        y2.append(y_next[0]*math.sin(y_next[1]))
        err_observe += math.sqrt((y_next[0]*math.cos(y_next[1])-x_next[0])**2+(y_next[0]*math.sin(y_next[1])-x_next[1])**2)
        err_EKF += math.sqrt((m_next[0]-x_next[0])**2+(m_next[1]-x_next[1])**2)
        m = m_next
        V = V_next
        x = x_next
    av_err_observe = err_observe/len(t-1)
    av_err_EKF = err_EKF/len(t-1)
    
    plt.plot(x1,x2,label='EKF_estimation',color='deeppink')
    plt.plot(x1_true,x2_true,label='actual',color='black')
    plt.xlim(-130,130)
    plt.ylim(-150,130)
    plt.legend(loc='lower right')
    plt.axes().set_aspect('equal','box')
    plt.show()
    
    plt.plot(y1,y2,label='observated_estimation',color='cyan')
    plt.plot(x1_true,x2_true,label='actual',color='black')
    plt.legend(loc='lower right')
    plt.xlim(-130,130)
    plt.ylim(-150,130)
    plt.axes().set_aspect('equal','box')
    plt.show()
    
    plt.plot(x1,x2,label='EKF_estimation',color='deeppink')
    plt.plot(y1,y2,label='observated_estimation',color='cyan')
    plt.plot(x1_true,x2_true,label='actual',color='black')
    plt.legend(loc='lower right')
    plt.xlim(-130,130)
    plt.ylim(-150,130)
    plt.axes().set_aspect('equal','box')
    plt.show()
    print('観測による推定値の誤差:{}'.format(av_err_observe))
    print('EKFによる推定値の誤差:{}'.format(av_err_EKF))
    
def main2():
    ##今回は使ってません。
    trial = 100
    err_observe_trans = []
    err_EKF_trans = []
    for step in range(trial):
        x0 = np.array([100,0])
        m0 = x0
        V0 = np.zeros((2,2))
        y0 = x0
    
        x = x0
        m = m0
        V = V0
        x1 = [x0[0]]
        x2 = [x0[1]]
        y1 = [y0[0]]
        y2 = [y0[1]]
        
        err_observe = 0
        err_EKF = 0
        for i in t-1:
            x_next,ui = get_next_x(i+1,x)
            y_next = observate_equation(x_next)
            m_next,V_next = EKF_One_Step(m,V,ui,y_next)
            x1.append(m_next[0])
            x2.append(m_next[1])
            y1.append(y_next[0]*math.cos(y_next[1]))
            y2.append(y_next[0]*math.sin(y_next[1]))
            err_observe += math.sqrt((y_next[0]*math.cos(y_next[1])-x_next[0])**2+(y_next[0]*math.sin(y_next[1])-x_next[1])**2)
            err_EKF += math.sqrt((m_next[0]-x_next[0])**2+(m_next[1]-x_next[1])**2)
            m = m_next
            V = V_next
            x = x_next
        av_err_observe = err_observe/len(t-1)
        av_err_EKF = err_EKF/len(t-1)
        err_observe_trans.append(av_err_observe)
        err_EKF_trans.append(av_err_EKF)
    plt.plot(range(trial),err_EKF_trans)
    plt.plot(range(trial),err_observe_trans)
    
    plt.show()

if __name__ == '__main__':
    main1()
    #main2()
        
    