# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:35:57 2018

@author: masaya
"""

import numpy as np ##いつもの
import numpy.random as rd ##ノイズ発生用
import numpy.linalg as lg ##いつもの
import math
import matplotlib.pyplot as plt##いつもの

#Constant
L = 100##半径
omega = math.pi/10##各加速度(tに対して一周するよう調整)
t = np.arange(21)##時刻
sigma_w = 3.0##外乱用
sigma_r = 10.0##観測ノイズ用
sigma_b = 5.0*math.pi/180##観測ノイズ用
R = np.array([[sigma_r**2,0],[0,sigma_b**2]])##観測ノイズ
Q = sigma_w**2*np.eye(2)##外乱ノイズ
A = np.array([[1,0],[0,1]])##fのヤコンビアン

def h(x):
    ##状態量xからh(x)を得る関数
    return np.array([math.sqrt(x[0]**2+x[1]**2),math.atan2(x[1],x[0])])

def get_next_x(i,x):
    ##時刻と前の状態量から、次の真値を求める関数。
    ##外乱wあり
    u = np.array([-L*omega*math.sin(omega*i),L*omega*math.cos(omega*i)])
    x_next = np.dot(A,x) + u +rd.normal(0.0,sigma_w,2)
    return x_next,u

def observate_equation(x):
    ##真値から得られる観測方程式
    ##ノイズvあり
    h_pure = h(x)
    y = h_pure + np.array([rd.normal(0.0,sigma_r),rd.normal(0.0,sigma_b)])
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

def EKF_One_Step(mt,Vt,ut,yt_plus_1):
    ##カルマンフィルタの生成アルゴリズム
    I = np.eye(len(Vt))#大きさ=vectorの単位行列
    ##予測ステップ
    m_prime = np.dot(A,mt)+ut##推定値
    h_m = h(m_prime)
    ##場合によってはatanが発散するので、例外処理
    if yt_plus_1[1]-h_m[1]>3:
        h_m[1] += 2*math.pi
    elif yt_plus_1[1]-h_m[1]<-3:
        h_m[1] -= 2*math.pi
    V_prime = np.dot(A,np.dot(Vt,A.T))+Q##推定共分散行列
    ##更新ステップ
    C = Ck(m_prime)
    K = np.dot(np.dot(V_prime,C.T),lg.inv(np.dot(np.dot(C,V_prime),C.T)+R))
    m_next = m_prime + np.dot(K , yt_plus_1 - h_m)##更新された推定値
    
    V_next = np.dot(I-np.dot(K,C),V_prime)##更新された推定共分散行列
    return m_next,V_next

def PF_One_Step(i,N,X,W,yt_plus_1):
    ##パーティクルフィルタの生成アルゴリズム
    W_next = np.zeros(N)
    W[0] =1.0/N
    X_next=np.zeros([N,2])
    for j in range(N):
        X_next[j],u = get_next_x(i,X[j])
        Y = h(X_next[j])
        if yt_plus_1[1]-Y[1]>3.0:
            Y[1] += 2.0*math.pi
        elif yt_plus_1[1]-Y[1]<-3.0:
            Y[1] -= 2.0*math.pi
        W_next[j] = W[j]*math.exp(-(yt_plus_1[0]-Y[0])**2/sigma_r**2-(yt_plus_1[1]-Y[1])**2/sigma_b**2)
    return W_next,X_next
        

def main1():
    ##点の推移
    ##観測・EKF・PF全てで行う
    N = 1000
    x0 = np.array([100,0])
    m0 = x0
    V0 = np.zeros((2,2))
    y0 = x0
    W0 = np.zeros(N)
    for j in range(N):
        W0[j] =1.0/N
    
    x = x0
    m = m0
    V = V0
    W = W0
    x1_true = [x0[0]]
    x2_true = [x0[1]]
    x1_PF = [x0[0]]
    x2_PF = [x0[1]]
    x1_EKF = [x0[0]]
    x2_EKF = [x0[1]]
    y1 = [y0[0]]
    y2 = [y0[1]]
    err_observe = 0
    err_EKF = 0
    err_PF = 0

    
    X0 = np.zeros([N,2])
    for j in range(N):
        X0[j] = np.array([rd.normal(100,0.1),rd.normal(0,0.1)])
    X = X0
    for i in t-1:
        x_next,ui = get_next_x(i+1,x)
        y_next = observate_equation(x_next)
        m_next,V_next = EKF_One_Step(m,V,ui,y_next)
        x1_EKF.append(m_next[0])
        x2_EKF.append(m_next[1])
        err_observe += math.sqrt((y_next[0]*math.cos(y_next[1])-x_next[0])**2+(y_next[0]*math.sin(y_next[1])-x_next[1])**2)
        err_EKF += math.sqrt((m_next[0]-x_next[0])**2+(m_next[1]-x_next[1])**2)
        m = m_next
        V = V_next
        err_observe += math.sqrt((y_next[0]*math.cos(y_next[1])-x_next[0])**2+(y_next[0]*math.sin(y_next[1])-x_next[1])**2)
        err_EKF += math.sqrt((m_next[0]-x_next[0])**2+(m_next[1]-x_next[1])**2)
        
        x1_true.append(x_next[0])
        x2_true.append(x_next[1])
        
        X_mean_t = np.array([0.0,0.0])
        W_next,X_next = PF_One_Step(i+1,N,X,W,y_next)
        W_next=W_next/sum(W_next)
        
        for j in range(N):
            X_mean_t[0] += W_next[j]*X[j,0]
            X_mean_t[1] += W_next[j]*X[j,1]
        err_PF += math.sqrt((X_mean_t[0]-x[0])**2+(X_mean_t[1]-x[1])**2)
        X = X_next
        W = W_next
        x = x_next
        x1_PF.append(X_mean_t[0])
        x2_PF.append(X_mean_t[1])
        y1.append(y_next[0]*math.cos(y_next[1]))
        y2.append(y_next[0]*math.sin(y_next[1]))
    
    plt.plot(x1_PF,x2_PF,label='PF',color='r')
    plt.plot(x1_true,x2_true,label='actual',color='black')
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.show()
    
    plt.plot(y1,y2,label='observed',color='cyan')
    plt.plot(x1_EKF,x2_EKF,label='EKF',color='deeppink')
    plt.plot(x1_PF,x2_PF,label='PF',color='r')
    plt.plot(x1_true,x2_true,label='actual',color='black')
    plt.legend(loc='lower right')
    plt.xlim(-130,130)
    plt.ylim(-150,130)
    plt.axis('equal')
    plt.show()
    av_err_observe = err_observe/len(t-1)
    av_err_EKF = err_EKF/len(t-1)
    av_err_PF = err_PF/len(t-1)
    print('観測による推定値の誤差:{}'.format(av_err_observe))
    print('EKFによる推定値の誤差:{}'.format(av_err_EKF))
    print('PFによる推定値の誤差:{}'.format(av_err_PF))

def main2():
    ##誤差計算用
    step = range(100)
    err_lst_observe=[]
    err_lst_EKF=[]
    err_lst_PF=[]
    for step_i in step:
        N = 1000
        x0 = np.array([100,0])
        m0 = x0
        V0 = np.zeros((2,2))
        y0 = x0
        W0 = np.zeros(N)
        for j in range(N):
            W0[j] =1.0/N
    
        x = x0
        m = m0
        V = V0
        W = W0
        err_observe = 0
        err_EKF = 0
        err_PF = 0
        X0 = np.zeros([N,2])
        for j in range(N):
            X0[j] = np.array([rd.normal(100,0.1),rd.normal(0,0.1)])
            X = X0
        for i in t-1:
            x_next,ui = get_next_x(i+1,x)
            y_next = observate_equation(x_next)
            m_next,V_next = EKF_One_Step(m,V,ui,y_next)
            err_observe += math.sqrt((y_next[0]*math.cos(y_next[1])-x_next[0])**2+(y_next[0]*math.sin(y_next[1])-x_next[1])**2)
            err_EKF += math.sqrt((m_next[0]-x_next[0])**2+(m_next[1]-x_next[1])**2)
            m = m_next
            V = V_next
            err_observe += math.sqrt((y_next[0]*math.cos(y_next[1])-x_next[0])**2+(y_next[0]*math.sin(y_next[1])-x_next[1])**2)
            err_EKF += math.sqrt((m_next[0]-x_next[0])**2+(m_next[1]-x_next[1])**2)
            X_mean_t = np.array([0.0,0.0])
            W_next,X_next = PF_One_Step(i+1,N,X,W,y_next)
            W_next=W_next/sum(W_next)
        
            for j in range(N):
                X_mean_t[0] += W_next[j]*X[j,0]
                X_mean_t[1] += W_next[j]*X[j,1]
            err_PF += math.sqrt((X_mean_t[0]-x[0])**2+(X_mean_t[1]-x[1])**2)
            X = X_next
            W = W_next
            x = x_next
        
        av_err_observe = err_observe/len(t-1)
        av_err_EKF = err_EKF/len(t-1)
        av_err_PF = err_PF/len(t-1)
        err_lst_observe.append(av_err_observe)
        err_lst_EKF.append(av_err_EKF)
        err_lst_PF.append(av_err_PF)
    plt.plot(step,err_lst_observe,label='err:observed',color='cyan')
    plt.plot(step,err_lst_EKF,label='err:EKF',color='deeppink')
    plt.plot(step,err_lst_PF,label='err:PF',color='r')
    plt.legend(loc='upper right')
    plt.xlabel('step')
    plt.ylabel('err-average')
    plt.show()
    
def main3():
    ##PF誤差計算用
    N_lst = np.linspace(10,10000,100)
    err_lst = []
    for N in N_lst:
        N=int(N)
        err_lst_PF=[]
        x0 = np.array([100,0])
        W0 = np.zeros(N)
        for j in range(N):
            W0[j] =1.0/N
    
        x = x0
        W = W0
        err_PF = 0
        X0 = np.zeros([N,2])
        for j in range(N):
            X0[j] = np.array([rd.normal(100,0.1),rd.normal(0,0.1)])
            X = X0
        for i in t-1:
            x_next,ui = get_next_x(i+1,x)
            y_next = observate_equation(x_next)
            X_mean_t = np.array([0.0,0.0])
            W_next,X_next = PF_One_Step(i+1,N,X,W,y_next)
            W_next=W_next/sum(W_next)
        
            for j in range(N):
                X_mean_t[0] += W_next[j]*X[j,0]
                X_mean_t[1] += W_next[j]*X[j,1]
            err_PF += math.sqrt((X_mean_t[0]-x[0])**2+(X_mean_t[1]-x[1])**2)
            X = X_next
            W = W_next
            x = x_next
       
        av_err_PF = err_PF/len(t-1)
        err_lst.append(av_err_PF)
    plt.plot(N_lst,err_lst,label='err:PF',color='r')
    plt.legend(loc='upper right')
    plt.xlabel('N')
    plt.ylabel('err-average')
    plt.show()

def main():
    ##統合
    ##重い時はそれぞれで計算した方が早いかも
    main1()
    main2()
    main3()

if __name__ =='__main__':
    main()      
        
        
        
        