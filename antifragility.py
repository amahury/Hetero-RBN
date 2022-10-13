# -*- coding: utf-8 -*-
"""
@author: Ama
"""
import mi_rbn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from  tqdm import tqdm
from scipy import stats

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    start_time = time.time()
    
    N = 100
    p = 0.5
    T = 100
    P = 5
    Q = 4
    X = 40
#    O = 1
    
    number_of_iterations=1000
    plt.title("Average fragility, HeSHeTHeF")
    plt.ylabel("Fragility")
#    plt.xlabel("X")
    plt.xlabel("O")
    colors=['b', 'orange', 'g', 'brown', 'purple', 'olive', 'gray', 'cyan', ]
    i=0
#    rango=np.arange(0, 100, 1.0) ##Range for X
    rango=np.arange(1, 51, 1.0) ##Range for O
    for (netType, mode, period, dist, K) in [
##            ("CRBN", "Complex", "", "Poisson", 4),
##            ("CRBN", "Complex", "", "Poisson", 5),
##            ("CRBN", "Complex", "", "Poisson", 3)
##            ("CRBN", "Complex", "", "Exponential", 1),
##            ("CRBN", "Complex", "", "Exponential", 2),
##            ("CRBN", "Complex", "", "Exponential", 3)
##            ("DGARBN", "Complex", "outdegree", "Poisson", 1),
##            ("DGARBN", "Complex", "outdegree", "Poisson", 2),
##            ("DGARBN", "Complex", "outdegree", "Poisson", 3)
##            ("DGARBN", "Complex", "outdegree", "Exponential", 1), #Stable network
            ("DGARBN", "Complex", "outdegree", "Exponential", 4), #Critical network
            ("DGARBN", "Complex", "outdegree", "Exponential", 5)  #Chaotic network
    ]:
        g1=[]
###        yerr=[]
        for O in tqdm(rango):
#        for X in tqdm(rango):
            print(netType, period, dist, K)
#            print(X)
            print(O)
            C0=[]
            C=[]
            for x in range(number_of_iterations):
                red=mi_rbn.RBN(N, float(K), p)
                param = ""
                if dist == "Exponential":
                    param = float(K)
                elif dist == "Zipf":
                    param = 2.0
                    
                if netType == "CRBN":
                    red.CreateNetCRBN(mode, distribution=dist, parameter=param)
                else:
                    red.CreateNetDGARBN(mode, P, Q, period, distribution=dist, parameter=param)

#        State0 is the dynamics without perturbation and State is the dynamics with perturbaciones
                if netType == "CRBN":
                    State0=red.RunNetCRBN(2*T, X=0, O=0)
                else:
                    State0=red.RunNetDGARBN(2*T, X=0, O=0)
                C0.append(np.mean(red.complexity(State0[-T:])))

#        O is given and X is changing, but we could reverse this                  
                if netType == "CRBN":
                    State=red.RunNetCRBN(2*T, X=int(X), O=int(O))
                else:
                    State=red.RunNetDGARBN(2*T, X=int(X), O=int(O))
                C.append(np.mean(red.complexity(State[-T:])))
                
                f = red.fragility(C=C, C0=C0, X=int(X), O=int(O), N=N, T=T)
            
            g1.append(np.mean(f))
###            yerr.append(f)
            
#        plt.errorbar(rango, g1, label="K="+K, yerr=stats.sem(yerr,1), ecolor='r', color=colors[i])
        plt.plot(rango, g1, label="K="+str(K), color=colors[i])
        i+=1   
        plt.ylim(top=1)
        np.savez("/home/amahury/fHeSHeTHeF_OK"+str(K)+".npz", g1) 
    plt.legend(prop={'size': 6}, loc='upper right')
    plt.savefig("/home/amahury/fHeSHeTHeF_O.png")
    print("--- %s seconds ---" % (time.time() - start_time))
    
