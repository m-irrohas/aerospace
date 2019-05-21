import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def gaussian(x,m,v):
  ##平均m,分散ｖとサンプルxからガウシアンを求める
  ##v = sigma^2で与えられる
  return np.exp(-(x-m)**2 / (2*v))/np.sqrt(2*np.pi*v)

def E_step(xs,ms,vs,p):
  #xsは計測値、ms,vsは平均,分散のリスト、pは男子の確率密度
  burden_rates = []
  for x in xs:
    d = (1-p) *gaussian(x,ms[0],vs[0]) + p*gaussian(x,ms[1],vs[1])
    n = p*gaussian(x,ms[1],vs[1])
    burden_rate = n/d
    burden_rates.append(burden_rate)
  
  return burden_rates

def M_step(xs,burden_rates):
  #xsのリストにたいして負担率があることに注意
  d1 = 0
  d2 = 0
  n1 = 0
  n2 = 0
  l1 = 0
  l2 = 0
  for x,r in zip(xs,burden_rates):
    d1 += 1-r
    d2 += r
    n1 += (1-r)*x
    n2 += r*x
  mu1 = n1/d1
  mu2 = n2/d2
  for x,r in zip(xs,burden_rates):
    l1 += (1-r)*(x-mu1)**2
    l2 += r*(x-mu2)**2
  var1 = l1/d1
  var2 = l2/d2

  N = len(xs)
  p = d2/N

  return mu1,mu2,var1,var2,p

def get_log_likelyhood(xs,ms,vs,p):
  #対数尤度を求める
  s = 0
  for x in xs:
    gauss1 = gaussian(x,ms[0],vs[0])
    gauss2 = gaussian(x,ms[1],vs[1])
    s += np.log((1-p)*gauss1+p*gauss2)
  return s

def main():
  #data
  data = [22.2,15.1,15.9,30.7,16.2,41.0,40.3,35.5,34.3,52.3,
  42.2,25.2,17.3,54.6,40.4,32.6,24.0,22.3,30.9,37.0,
  21.1,23.7,19.7,21.0,16.4,34.3,18.1,26.8,12.1,7.3,
  15.5,10.7,59.4,30.5,8.0,44.0,27.3,16.0,12.4,18.1,
  18.1,34.2,10.3,33.8,34.2,0.8,32.2,11.8,26.1,24.1]

  #initial state
  p = 0.5
  ms = [random.choice(data),random.choice(data)]
  vs = [np.var(data),np.var(data)]
  ##試行回数
  T = 1000
  ##保存データ
  ls = []
  
  ##以下EMアルゴリズム
  for t in range(T):
    burden_rates = E_step(data,ms,vs,p)
    mu1,mu2,var1,var2,p = M_step(data,burden_rates)
    ms = [mu1,mu2]
    vs = [var1,var2]
    ls.append(get_log_likelyhood(data,ms,vs,p))

  print("mu1={0},mu2={1}\n v1={2},v2={3}\n pi_F={4},pi_M={5}".format(ms[0],ms[1],vs[0],vs[1],1-p,p))

  #plot
  x_plot = np.linspace(min(data),max(data),200)
  n = len(data)
  man = np.zeros_like(x_plot)
  female = np.zeros_like(man)
  mix = np.zeros_like(man)
  for i in range(len(x_plot)):
    man[i] = n*p*gaussian(x_plot[i],ms[1],vs[1])
    female[i] = n*(1-p)*gaussian(x_plot[i],ms[0],vs[0])
    mix[i] = man[i]+female[i]

  ##ヒストグラム
  plt.hist(data,label="histgram")
  plt.plot(x_plot,6*man,label="Man")
  plt.plot(x_plot,6*female,label="Female")
  plt.plot(x_plot,6*mix,label="MIX")
  plt.legend()
  plt.xlim(0,60)
  plt.ylim(0,12)
  plt.show()

  plt.subplot(211)
  norm1 = mlab.normpdf(x_plot,ms[0],np.sqrt(vs[0]))
  norm2 = mlab.normpdf(x_plot,ms[1],np.sqrt(vs[1]))
  plt.plot(x_plot,(1-p)*norm1+p*norm2,color='red',lw=3)
  plt.xlim(min(data),max(data))
  plt.xlabel("x[m]")
  plt.ylabel("Probability[]")
  
  plt.subplot(212)
  plt.plot(np.arange(len(ls)),ls)
  plt.xlabel("step[Number]")
  plt.ylabel("log_likelihood")
  plt.show()
if __name__ == '__main__':
  main()
