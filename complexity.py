# -*- coding: utf-8 -*-
"""
@author: okarim
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
    
    N=100
    p=0.5
    T=1000
    P = 5
    Q = 4
    
    number_of_iterations=1000
    plt.title("Complexity")
    plt.ylabel("Complexity")
    plt.xlabel("K")
    colors=['b', 'orange', 'g', 'brown', 'purple', 'olive', 'gray', 'pink', ]
    i=0
    rango=np.arange(1.1, 10.1, 0.2) #Por probar dK=0.5 y dK=1.0 
    for (netType, mode, period, dist) in [
#           ("DGARBN", "Random"),
#            ("Random", "outdegree"), ("Random", "ceil"),
##            ("CRBN", "Complex", "", "Poisson"),
##            ("CRBN", "Complex", "", "Exponential"),
#            ("CRBN", "Complex", "", "Zipf"),
            #("Complex", "ceil"),
            #("Complex", ""),
            # --------------------
#            ("DGARBN", "Random", "", "Poisson"),
#            ("DGARBN", "Complex", "", "Zipf"),
#            ("DGARBN", "Complex", "", "Exponential"),
            # --------------------
##            ("DGARBN", "Complex", "outdegree", "Poisson"),
#            ("DGARBN", "Complex", "outdegree", "Zipf"),
            ("DGARBN", "Complex", "outdegree", "Exponential")
    ]:
        g1=[]
        yerr=[]
        for K in tqdm(rango):
            print(netType, mode, dist)
            print(K)
            C=[]
            for x in range(number_of_iterations):
                red=mi_rbn.RBN(N, float(K), p)
                param = ""
                if dist == "Exponential":
                    param = float(K)
                elif dist == "Zipf":
                    param = 2.0
                    
                if netType == "CRBN":
#                    red.CreateNetCRBN(mode)
                    red.CreateNetCRBN(mode, distribution=dist, parameter=param)
                else:
                    red.CreateNetDGARBN(mode, P, Q, period, distribution=dist, parameter=param)

                    
                if netType == "CRBN":
                    State=red.RunNetCRBN(2*T)
                else:
                    State=red.RunNetDGARBN(2*T)
                C.append(np.mean(red.complexity(State[-T:])))

            g1.append(np.mean(C))
            yerr.append(C)
    
        plt.errorbar(rango, g1, label=""+netType+" "+period + " - " + dist, yerr=stats.sem(yerr,1), ecolor='r', color=colors[i])
#        plt.plot(rango, g1, label=""+netType+" "+dist)
        i+=1
#        plt.title("Average Complexity")    
        plt.ylim(top=1)
        np.savez("/home/amahury/SI10_10.npz", g1)
    plt.legend(prop={'size': 6}, loc='upper right')
    plt.savefig("/home/amahury/SI10_10.png")
#    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
