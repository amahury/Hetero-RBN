import time
import matplotlib.pyplot as plt
#import multiprocessing
#from functools import partial
import numpy as np
#import networkx as nx # this is just for drawing the graph
#from networkx.drawing.nx_agraph import to_agraph 

class RBN:
    def __init__(self, N, K, p):
        """
        K = number of connections
        N = number of nodes, indexed 0 .. N-1
        p = probability of one within rules
        """
        self.K=K
        self.N=N
        self.p=p
        self.netType = ""


    def gaussian(self, low, high, mean, sigma, M):
        v = []
        while len(v) < M:
            x = np.random.normal(mean, sigma)
            if low <= x <= high:
                v.append(x)
        norm = lambda t: t / np.amax(v)
        return norm(v)

        
    def CreateNetDGARBN(self, topology, P, Q, period_type="", distribution="Zipf", parameter=0, degree_sequence=[]):
        """
        P = maximum period in node activation in DGARBNs.
        Q = maximum probability of translation in node activation in DGARBNs.
        period_type = outdegree, ceil or none
        """
        self.CreateNetCRBN(topology, distribution, parameter, degree_sequence)
        self.netType = "DGARBN"
        if period_type == "outdegree":
            # el numero de salidas del nodo = Pi
            # entre más conexiones, mas lento se actualiza el nodo
            if topology == "Complex":
                self.periods = np.zeros(self.N+1) #create periods for DGARBN
                for i in range(self.N+1):
                    self.periods[i] = np.count_nonzero(self.Con[i])
                self.transitions = np.random.randint(0,Q+1, size=self.N+1) #create translations for DGARBN
                self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
            else:
                if(type(self.K) is int):
                    self.periods = np.full(self.N, self.K) #create periods for DGARBN
                    self.transitions = np.random.randint(0,Q+1, size=self.N) #create translations for DGARBN
                    self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
                else:
                    self.periods = np.zeros(self.N+1) #create periods for DGARBN
                    for i in range(self.N+1):
                        self.periods[i] = np.count_nonzero(self.Con[i])
                    self.transitions = np.random.randint(0,Q+1, size=self.N+1) #create translations for DGARBN
                    self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
        elif period_type == "ceil":
            # misma P para todos
            # techo de la K actual
            if topology == "Complex":
                p = np.ceil(self.actual_k)
                self.periods = np.full(self.N+1, p) #create periods for DGARBN
                self.transitions = np.random.randint(0,Q+1, size=self.N+1) #create translations for DGARBN
                self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
            else:
                if(type(self.K) is int):
                    self.periods = np.full(self.N, self.actual_k) #create periods for DGARBN
                    self.transitions = np.random.randint(0,Q+1, size=self.N) #create translations for DGARBN
                    self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
                else:
                    p = np.ceil(self.actual_k)
                    self.periods = np.full(self.N+1, p) #create periods for DGARBN
                    self.transitions = np.random.randint(0,Q+1, size=self.N+1) #create translations for DGARBN
                    self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
        else:
            if topology == "Complex":
                self.periods = np.random.randint(0,P+1,size=self.N+1) #create periods for DGARBN
                self.transitions = np.random.randint(0,Q+1, size=self.N+1) #create translations for DGARBN
                self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
            else:
                if(type(self.K) is int):
                    self.periods = np.random.randint(0,P+1,size=self.N) #create periods for DGARBN
                    self.transitions = np.random.randint(0,Q+1, size=self.N) #create translations for DGARBN
                    self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
                else:
                    self.periods = np.random.randint(0,P+1,size=self.N+1) #create periods for DGARBN
                    self.transitions = np.random.randint(0,Q+1, size=self.N+1) #create translations for DGARBN
                    self.transitions[np.where(self.periods<=self.transitions)]=np.random.randint(0,Q)
        self.periods = self.periods.astype(int)
             

    def CreateNetCRBN(self, topology, distribution="Zipf", parameter=2.0, degree_sequence=[]):
        """
        p = probability of one within rules
        topology = topology of the Net: Random or Configuration Model
        distribution = distribution of probability
        parameter = parameter for the probability function used in topology
        """
        self.netType = "CRBN"
        if topology == "Complex":
            self.K = float(self.K)
            self.ConfigurationModel(distribution, parameter, degree_sequence)
            self.actual_k = self.k_mean()
        else:
            if(type(self.K) is int):
                self.Con = np.apply_along_axis(np.random.permutation, 1, np.tile(range(self.N), (self.N,1) ))[:, 0:self.K]
                self.Bool = np.random.choice([0, 1], size=(self.N, 2**self.K), p=[1-self.p, self.p]) # N random boolean functions, a list of 2^k  ones and zeros.
            else:
                if degree_sequence == []:
                    Kv=np.random.poisson(self.K, self.N)
                    Kv[np.where(Kv>self.N)]=self.N
                    Kv[np.where(Kv==0)]=1
                else:
                    Kv = degree_sequence
                maximo=np.amax(Kv)

                self.Con=np.zeros((self.N+1, maximo),dtype=int)
                self.Bool=np.zeros((self.N+1, 2**maximo),dtype=int)
                for i in range(self.N):
                    self.Con[i+1, 0:Kv[i]] = np.random.choice(self.N, Kv[i], replace=False)+1
                    self.Bool[i+1, 0:2**Kv[i]] = (np.random.choice([0, 1], size=2**Kv[i], p=[1-self.p, self.p]))
                self.actual_k = self.k_mean()

        return




    def ConfigurationModel(self, distribution, parameter, degree_sequence=[]):
        """
        distribution = distribution of probability
        parameter = parameter for the probability function used in topology
        """
        if degree_sequence == []:
            degree=[] # start with empty list
            if distribution=="Zipf":
#                degree = zipf.rvs(parameter, size=N)
#                degree = np.random.zipf(parameter, size=self.N)
#                degree[np.where(degree > 20)] = 20
                k=-np.inf
                while (k < self.K - 0.5 or k > self.K + 0.5 ):
                    degree = np.random.zipf(parameter, size=self.N)
#                    degree[np.where(degree > 20)] = 20
                    degree[np.where(degree > self.N)] = self.N
                    k=np.mean(degree)
            elif distribution=="Exponential":
                k=-np.inf
                while (k < self.K - 0.5 or k > self.K + 0.5 ):
                    degree = np.rint(np.random.exponential(parameter, size=self.N)).astype(int)
                    degree[np.where(degree > 25)] = 25
#                    degree[np.where(degree > self.N)] = self.N
                    k=np.mean(degree)
            else:  #poisson
                k=-np.inf
                while (k < self.K - 0.5 or k > self.K + 0.5 ):
                    degree=np.random.poisson(self.K, self.N)
                    degree[np.where(degree>self.N)]=self.N
                    degree[np.where(degree==0)]=1
                    k=np.mean(degree)
        else:
            degree = degree_sequence

        m=max(degree)
        self.Con=np.zeros((self.N+1, m),dtype=int)
        self.Bool=np.zeros((self.N+1, 2**m),dtype=int)
        probability = np.random.triangular(0.0, self.p, 1.0, self.N)
#        probability = np.random.uniform(low=0.0, high=1.0, size=self.N)
#        probability = self.gaussian(low=0.0, high=1.0, mean=self.p, sigma=1.0, M=self.N)
        for i in range(self.N):
            self.Con[i+1, 0:degree[i]] = np.random.choice(self.N, degree[i], replace=False)+1
            self.Bool[i+1, 0:2**degree[i]] = (np.random.choice([0, 1], size=2**degree[i], p=[1-probability[i], probability[i]])) 


                        
    def RunNetDGARBN(self, T, initial=[], X=0, O=0):
        """
        Con= matrix of connections
        Bool= lookup table
        T = timesteps
        initial = initial state (random if empty)
        M = how many perturbations
        O = how often the perturbations take place
        X = how many perturbations
        O = how often the perturbations take place
        """
        Pow = 2**np.arange(np.size(self.Con, 1)) # [ 1 2 4 ... ], for converting inputs to numerical value
        
        if(type(self.K) is int):
            a=0
            State = np.zeros((T+1,self.N),dtype=int)
            if np.array_equal(initial, []):
                State[0] = np.random.randint(0, 2, self.N) 
            else:
                State[0] = initial
        else:
            a=1
            State = np.zeros((T+1,self.N+1),dtype=int)
            if np.array_equal(initial, []):
                State[0] = np.append([0], np.random.randint(0, 2, self.N))
            else:
                State[0] = np.append([0],initial)

#            if self.Con.size != 0:
            self.Bool[np.where(self.Con[:,0]==0),0] = State[0, np.where(self.Con[:,0]==0)] # if node doesn't have conections not change


        # Update deterministically semi-synchronously for DGARBN
        for t in range(T):  # 0 .. T-1
            self.Bool[np.where(self.Con[:,0]==0),0] = State[t, np.where(self.Con[:,0]==0)]
            with np.errstate(divide='ignore'):
                b = (t % self.periods == self.transitions)
                
            tr = self.Bool[:, np.sum(Pow * State[t,self.Con],1)].diagonal()
            State[t+1] = np.where(b, tr, State[t])                    
            if ( X and O ) != 0:  #Perturbations
                if t%O == 0:
                    State[t+1,  np.random.choice(self.N, size=X, replace=False)+a] = np.random.randint(0, 2, X)
                                            
        if(type(self.K) is int):
            return(State)
        else:
            return(State[:,1:])

        
        
    def RunNetCRBN(self, T, initial=[], X=0, O=0):
        """
        Con= matrix of connections
        Bool= lookup table
        T = timesteps
        initial = initial state (random if empty)
        X = how many perturbations
        O = how often the perturbations take place
        """
        Pow = 2**np.arange(np.size(self.Con, 1)) # [ 1 2 4 ... ], for converting inputs to numerical value
        
        if(type(self.K) is int):
            a=0
            State = np.zeros((T+1,self.N),dtype=int)
            if np.array_equal(initial, []):
                State[0] = np.random.randint(0, 2, self.N) 
            else:
                State[0] = initial
        else:
            a=1
            State = np.zeros((T+1,self.N+1),dtype=int)
            if np.array_equal(initial, []):
                State[0] = np.append([0], np.random.randint(0, 2, self.N))
            else:
                State[0] = np.append([0],initial)

#            if self.Con.size != 0:
            self.Bool[np.where(self.Con[:,0]==0),0] = State[0, np.where(self.Con[:,0]==0)] # if node doesn't have conections not change

        for t in range(T):  # 0 .. T-1
            self.Bool[np.where(self.Con[:,0]==0),0] = State[t, np.where(self.Con[:,0]==0)]
            State[t+1] = self.Bool[:, np.sum(Pow * State[t,self.Con],1)].diagonal()
            if ( X and O ) != 0:  #Perturbations
                if t%O == 0:
                    State[t+1,  np.random.choice(self.N, size=X, replace=False)+a] = np.random.randint(0, 2, X)
                                            
        if(type(self.K) is int):
            return(State)
        else:
            return(State[:,1:])

        
    def antifragile_iterative(self, T, runs=1, X=None, O=None, fraction=1):
        f=np.zeros(int(self.N/fraction))
        
        for j in range(runs):
            initial = np.random.randint(0, 2, self.N)
            if self.netType == "CRBN":
                State=self.RunNetCRBN(2*T, initial)
            else:
                State=self.RunNetDGARBN(2*T, initial)
            C0 = self.complexity(State[-T:])
            if(O!=None):
                for i in range(1, int(self.N/fraction)+1):
                    f[i-1]+=self.func(i, T=T, initial=initial, O=O, C0=C0, fraction=fraction)
            elif(X!=None):
                for i in range(1, int(T/fraction)+1):
                    f[i-1]+=self.func2(i, T=T, initial=initial, X=X, C0=C0, fraction=fraction)
        f/=runs # average fragility by perturbation
        return f

        
#    def antifragile(self, T, runs=1, X=None, O=None, fraction=1):
#        f=np.zeros(int(self.N/fraction))
#        pool = multiprocessing.Pool()

#        for j in range(runs):
#            initial = np.random.randint(0, 2, self.N)
#            if self.netType == "CRBN":
#                State=self.RunNetCRBN(2*T, initial)
#            else:
#                State=self.RunNetDGARBN(2*T, initial)
#            C0 = self.complexity(State[-T:])
#            if(O!=None):
#                f+=pool.map(partial(self.func, T=T, initial=initial, O=O, C0=C0, fraction=fraction), range(1, int(self.N/fraction)+1))
#            elif(X!=None):
#                f+=pool.map(partial(self.func2, T=T, initial=initial, X=X, C0=C0, fraction=fraction), range(1, int(T/fraction)+1))
#        f/=runs # average fragility by perturbation
#        pool.close()
#        return f

    def func2(self, i, T, initial, X, C0, fraction=1):
        f=np.zeros(int(self.N/fraction))
        if self.netType == "CRBN":
            State=self.RunNetCRBN(2*T, initial, X, i)
        else:
            State=self.RunNetDGARBN(2*T, initial, X, i)
        C = self.complexity(State)
        f=self.fragility(C, C0, X, i, self.N, T)
        return f

    def func(self, X, T, initial, O, C0, fraction=1):
        if self.netType == "CRBN":
            State=self.RunNetCRBN(2*T, initial, X, O)
        else:
            State=self.RunNetDGARBN(2*T, initial, X, O)
        C = self.complexity(State)
        f=self.fragility(C, C0, X, O, self.N, T)
        return f

    def fragility(self, C, C0, X, O, N, T):
        """
        C0 = initial complexity
        C = final complexity
        X = how many perturbations
        O = how often the perturbations take place
        N = number of nodes, indexed 0 .. N-1
        T = timesteps
        """
        dx =(X*(T/O))/(N*T) # degree of perturbation
        sigma = np.mean(C)-np.mean(C0) # degree of satisfaction
        return -sigma*dx

                    
    def Attractors(self, topology, T, runs=0):
        """
        List of Attractors of R random initial states
        runs = number of runs (if 0 then List of Attractors of every possible initial state)
        T = timesteps
        """
        attractList=[]
        if runs == 0 :
            for i in range(np.power(2,self.N)):
                initial=[x=='1' for x in format(i, '0'+str(self.N)+'b')]
                if topology == "Complex":
                    State=self.RunNetDGARBN(T, initial)
                    # print("State")
                    # print(State)
                
                    # unique_elements, counts_elements = np.unique(State, return_counts=True, axis=0)                        
                    # A=unique_elements[np.where(counts_elements > 1)] #States that appear more than one occasion
                    # print(A)
                    # print("-------------")
                    
                    # ### CHECK PERIODS
                    # for x in A:
                    #     print(x)
                    #     t = np.where((State == x).all(axis=1))
                    #     print(t[0])

                    #     vf = np.zeros(shape=(len(self.periods),len(t[0])), dtype=int)
                    #     for j in range(len(self.periods)):
                    #         vf[j] = t[0] % self.periods[j]
                    #     print(vf)
                    #     print(vf.T)
                    #     vf = vf.T
                    #     print("################")

                    #     unique_elements, counts_elements = np.unique(vf, return_counts=True, axis=0)
                    #     print(unique_elements)
                    #     C=unique_elements[np.where(counts_elements > 1)] #States that appear more than one occasion
                    #     print(C)
                    #     print("$$$$$$$$$$$$$$")
                    
                    #     if not(C.tolist() in attractList):  #if A is not in attractList then add it
                    #         print("**** ENTRE ******")
                    #         print(C)
                    #         print(counts_elements)
                    #         attractList.append(C.tolist())
                    #         print(attractList)


                else:
                    State=self.RunNetCRBN(T, initial)

#                print("State")
#                print(State)
                
                unique_elements, counts_elements = np.unique(State, return_counts=True, axis=0)
                A=unique_elements[np.where(counts_elements > 1)] #States that appear more than one occasion
#                print(A)
#                print("-------------")
                    
                ### CHECK PERIODS
                for x in A:
#                    print(x)
                    t = np.where((State == x).all(axis=1))
#                    print(t[0])

                    vf = np.zeros(shape=(len(self.periods),len(t[0])), dtype=int)
                    for j in range(len(self.periods)):
                        vf[j] = t[0] % self.periods[j]
#                    print(vf)
#                    print(vf.T)
                    vf = vf.T
#                    print("################")

                    unique_elements, counts_elements = np.unique(vf, return_counts=True, axis=0)
#                    print(unique_elements)
                    C=unique_elements[np.where(counts_elements > 1)] #States that appear more than one occasion
#                    print(C)
#                    print("$$$$$$$$$$$$$$")
                    
                    if not(C.tolist() in attractList):  #if A is not in attractList then add it
#                        print("**** ENTRE ******")
#                        print(C)
#                        print(counts_elements)
                        attractList.append(C.tolist())
#                        print(attractList)

#                if not(A.tolist() in attractList):  #if A is not in attractList then add it
#                        attractList.append(A.tolist())

                    
        else:
            for i in range(runs):
                if topology == "Complex":
                    State=self.RunNetDGARBN(T)
                else:
                    State=self.RunNetCRBN(T)
                unique_elements, counts_elements = np.unique(State, return_counts=True, axis=0)      
                A=unique_elements[np.where(counts_elements > 1)] #States that appear more than one occasion
    
                if not(A.tolist() in attractList):  #if A is not in attractList then add it
                    attractList.append(A.tolist())
                
        return attractList


    def MeanAttractors(self, attractorsList):
        """
        Longitud promedio de Atractores
        """
        edos = 0
        for i in attractorsList:
            edos += len(i)
        return (edos/len(attractorsList))
            
        
    def set(self, K, N, T, p, P, Q, Con, Bool, periods, transitions):
        self.K = K
        self.N = N
        self.T = T
        self.p = p
        self.P = P
        self.Q = Q
        self.Con = Con
        self.Bool = Bool
        self.transitions = transitions
        self.periods = periods


#    def drawNetwork(self, path):
#        plt.figure()
#        G=nx.MultiDiGraph()
#        G.add_nodes_from(range(self.N))
#        for l in range(self.N):
#            for e in self.Con[l]:
#                if e != -1:
#                    G.add_edge(l,e)

#        A = to_agraph(G) 
#        A.layout('dot')
#        A.draw(path)
#        nx.draw(G,node_size=30, with_labels=True)
#        plt.show()


    def plot(self, State, model, distribution, complexi, real_K, path=""):
#        if distribution.startswith("Configuration"):
#            title = "Model:"+model + ", Distribution:"+distribution + ", K="+str(real_K)
#        else:
        title = "Model:"+model + ", Distribution:"+distribution + ", K="+str(round(real_K, 2))
        cp = "Complexity: "+str(np.round(np.mean(complexi), 4))
        plt.figure()
        plt.imshow(State, cmap='Greys', interpolation='None')
        plt.title(title)
        plt.gcf().text(0.05, 0.05, cp, bbox={'facecolor':'white', 'pad':2})
        plt.xlabel("# nodes")
        plt.ylabel("Iterations")
        plt.savefig(path)
#        plt.show()


    
    def RBNSort(self): 
        """
        Sort the nodes by their overall activity
        """
        SRun = 5     # sorting runs
        ST = 200     # sorting timesteps
        Totals = np.zeros(self.N,dtype=int)
        
        for r in range(SRun):
            State=self.RunNet(ST)
            Totals = Totals + np.sum(State, 0)
        
        Index = np.argsort(Totals)    # permutation indexes for sorted order
        
        if(type(self.K) is int):
            self.Bool = self.Bool[Index]         # permute the boolean functions
            self.Con = self.Con[Index]           # permute the connections
            
            InvIndex = np.argsort(Index)         # inverse permutation
            self.Con = InvIndex[self.Con]        # relabel the connections
        else:
            self.Bool[1:] = self.Bool[Index+1]         # permute the boolean functions
            self.Con[1:] = self.Con[Index+1]           # permute the connections
            InvIndex = np.append([-1], np.argsort(Index)) # inverse permutation
            self.Con[1:] = InvIndex[self.Con[1:]]+1        # relabel the connections
        return



    def complexity(self, state):
        """
        Measuring Complexity Based on Shanon Entropy 
        state = matrix of a RBN states
        """
        p1=np.sum(state, axis=0)/np.size(state, 0)
        p0=1-p1
        np.place(p0, p0==0, 1)
        np.place(p1, p1==0, 1)

        #column by column
        E=-(p0*np.log2(p0)+p1*np.log2(p1)) #Shannon Entropy
        E=np.mean(E)
        C=4*E*(1-E) #Complexity
        return C


    def k_mean(self):
        return np.count_nonzero(self.Con)/self.N


def plotLenAttractorsRandom(k1, k2, k3, k4, cp1, cp2, cp3, cp4, ylabel, path):
    plt.figure()
    plt.title("Attractors")
    plt.xlabel("K")
    plt.ylabel(ylabel)
#    plt.ylabel("Number of Attractors")
    plt.plot(k1, cp1, label="CRBN-Random")
    plt.plot(k2, cp2, label="DGARBN-Random")
    plt.plot(k3, cp3, label="DGARBN-Random Outdegree")
    plt.plot(k4, cp4, label="DGARBN-Random Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)
                                        

def plotLenAttractorsConfiguration(k5, k6, k7, k8, cp5, cp6, cp7, cp8, ylabel, path):
    plt.figure()
    plt.title("Attractors")
    plt.xlabel("K")
    plt.ylabel(ylabel)
#    plt.ylabel("Number of Attractors")
    plt.plot(k5, cp5, label="CRBN-Configuration Model")
    plt.plot(k6, cp6, label="DGARBN-Configuration Model")
    plt.plot(k7, cp7, label="DGARBN-Configuration Model Outdegree")
    plt.plot(k8, cp8, label="DGARBN-Configuration Model Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)

                                        
def plotAttractorsRandom(k1, k2, k3, k4, cp1, cp2, cp3, cp4, ylabel, path):
    plt.figure()
    plt.title("Attractors Measures")
    plt.xlabel("K")
    plt.ylabel(ylabel)
#    plt.ylabel("Attractors Average Length")
    plt.plot(k1, cp1, label="CRBN-Random")
    plt.plot(k2, cp2, label="DGARBN-Random")
    plt.plot(k3, cp3, label="DGARBN-Random Outdegree")
    plt.plot(k4, cp4, label="DGARBN-Random Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)

    
def plotAttractorsConfiguration(k5, k6, k7, k8, cp5, cp6, cp7, cp8, ylabel, path):
    plt.figure()
    plt.title("Attractors Measures")
    plt.xlabel("K")
#    plt.ylabel("Attractors Average Length")
    plt.ylabel(ylabel)
    plt.plot(k5, cp5, label="CRBN-Configuration Model")
    plt.plot(k6, cp6, label="DGARBN-Configuration Model")
    plt.plot(k7, cp7, label="DGARBN-Configuration Model Outdegree")
    plt.plot(k8, cp8, label="DGARBN-Configuration Model Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)

                
def plotComplexitiesRandom(k1, k2, k3, k4, cp1, cp2, cp3, cp4, path):
    plt.figure()
    plt.title("Complexity Measures")
    plt.xlabel("K")
    plt.ylabel("Complexity")
    plt.plot(k1, cp1, label="CRBN-Random")
    plt.plot(k2, cp2, label="DGARBN-Random")
    plt.plot(k3, cp3, label="DGARBN-Random Outdegree")
    plt.plot(k4, cp4, label="DGARBN-Random Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)

    
def plotComplexitiesConfiguration(k5, k6, k7, k8, cp5, cp6, cp7, cp8, path):
    plt.figure()
    plt.title("Complexity Measures")
    plt.xlabel("K")
    plt.ylabel("Complexity")
    plt.plot(k5, cp5, label="CRBN-Configuration Model")
    plt.plot(k6, cp6, label="DGARBN-Configuration Model")
    plt.plot(k7, cp7, label="DGARBN-Configuration Model Outdegree")
    plt.plot(k8, cp8, label="DGARBN-Configuration Model Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)

def plotAntifragilityConfiguration(k5, cp5, title, path):
    plt.figure()
    plt.title("Difference in Complexity\n"+title)
    plt.xlabel("X")
    plt.ylabel(r"$\phi$")
#    plt.ylabel(u"\u0394\u03C3")
    x = np.arange(1,len(k5)+1)
    for i in range(len(cp5)):
        plt.plot(cp5[i], label="K="+str(k5[i]))
    plt.legend(loc='best',prop={'size': 9})
    plt.savefig(path)
    
def plotK(k5, k6, k7, k8, cp5, cp6, cp7, cp8, path):
    plt.figure()
    plt.title("K-Measures")
    plt.xlabel("log k")
    plt.ylabel("log real K")
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(k5, cp5, label="CRBN-Configuration Model")
    plt.plot(k6, cp6, label="DGARBN-Configuration Model")
    plt.plot(k7, cp7, label="DGARBN-Configuration Model Outdegree")
    plt.plot(k8, cp8, label="DGARBN-Configuration Model Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)
                                        
def plotP(k5, k6, k7, k8, cp5, cp6, cp7, cp8, path):
    plt.figure()
    plt.title("P-Measures")
    plt.xlabel("log k")
    plt.ylabel("log P")
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(k5, cp5, label="CRBN-Configuration Model")
    plt.plot(k6, cp6, label="DGARBN-Configuration Model")
    plt.plot(k7, cp7, label="DGARBN-Configuration Model Outdegree")
    plt.plot(k8, cp8, label="DGARBN-Configuration Model Ceil")
    plt.legend(loc='best',prop={'size': 8})
    plt.savefig(path)
                                                
    
def testComplexity(N, K, p, T, P, Q, distribution, parameter, path, attractorRuns=1000, plots=True):
#    initial = np.random.randint(0, 2, N)
#    initial = []
#    red=RBN(N, K, p)
#    Kv=np.random.poisson(K, N)
#    Kv[np.where(Kv>N)]=N

    print("----Testing CRBN-Random---"+str(K))
#    red.CreateNetCRBN("Random", degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetCRBN("Random")
#    print("Attractors")
#    att1 = red.Attractors("Random", T, runs=attractorRuns)
#    a1 = red.MeanAttractors(att1)
    print("Running")
    State = red.RunNetCRBN(T, initial)
    cp1 = red.complexity(State)
    new_k1 = red.actual_k

    print("----Testing DGARBN-Random----"+str(K))
#    red.CreateNetDGARBN("Random", P, Q, degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetDGARBN("Random", P, Q, parameter=K)
#    print("Attractors")
#    att2 = red.Attractors("Random", T, runs=attractorRuns)
#    a2 = red.MeanAttractors(att2)
    print("Running")
    State = red.RunNetDGARBN(T, initial)
    cp2 = red.complexity(State)
    new_k2 = red.actual_k
    
    print("----Testing DGARBN-Random-Outdegree----"+str(K))
#    red.CreateNetDGARBN("Random", P, Q, degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetDGARBN("Random", P, Q, "outdegree", parameter=K)
#    print("Attractors")
#    att3 = red.Attractors("Random", T, runs=attractorRuns)
#    a3 = red.MeanAttractors(att3)
    print("Running")
    State = red.RunNetDGARBN(T, initial)
    cp3 = red.complexity(State)
    new_k3 = red.actual_k

    print("----Testing DGARBN-Random-ceil----"+str(K))
#    red.CreateNetDGARBN("Random", P, Q, degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetDGARBN("Random", P, Q, "ceil", parameter=K)
#    print("Attractors")
#    att4 = red.Attractors("Random", T, runs=attractorRuns)
#    a4 = red.MeanAttractors(att4)
    print("Running")
    State = red.RunNetDGARBN(T, initial)
    cp4 = red.complexity(State)
    new_k4 = red.actual_k
    
    print("----Testing CRBN-Complex---"+str(K))
#    red.CreateNetCRBN("Complex", "Zipf", 2.3, degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetCRBN("Complex", "Zipf", parameter=K)
#    print("Attractors")
#    att5 = red.Attractors("Random", T, runs=attractorRuns)
#    a5 = red.MeanAttractors(att5)
    print("Running")
    State = red.RunNetCRBN(T, initial)
    cp5 = red.complexity(State)
    new_k5 = red.actual_k

    print("----Testing DGARBN-Complex----"+str(K))
#    red.CreateNetDGARBN("Complex", P, Q, "Zipf", 2.3, degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetDGARBN("Complex", P, Q, "Zipf", parameter=K)
#    print("Attractors")
#    att6 = red.Attractors("Complex", T, runs=attractorRuns)
#    a6 = red.MeanAttractors(att6)
    print("Running")
    State = red.RunNetDGARBN(T, initial)
    cp6 = red.complexity(State)
    new_k6 = red.actual_k
    
    print("----Testing DGARBN-Complex-outdegree----"+str(K))
#    red.CreateNetDGARBN("Complex", P, Q, "Zipf", 2.3, degree_sequence=Kv)
    print("Constructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetDGARBN("Complex", P, Q, "outdegree", "Zipf", parameter=K)
#    print("Attractors")
#    att7 = red.Attractors("Complex", T, runs=attractorRuns)
#    a7 = red.MeanAttractors(att7)
    print("Running")
    State = red.RunNetDGARBN(T, initial)
    cp7 = red.complexity(State)
    new_k7 = red.actual_k

    print("----Testing DGARBN-Complex-ceil----"+str(K))
#    red.CreateNetDGARBN("Complex", P, Q, "Zipf", 2.3, degree_sequence=Kv)
    print("Contructing")
    initial = []
    red=RBN(N, K, p)
    red.CreateNetDGARBN("Complex", P, Q, "ceil", "Zipf", parameter=K)
#    print("Attractors")
#    att8 = red.Attractors("Complex", T, runs=attractorRuns)
#    a8 = red.MeanAttractors(att8)
    print("Running")
    State = red.RunNetDGARBN(T, initial)
    cp8 = red.complexity(State)
    new_k8 = red.actual_k

    
    if plots:
        red.plot(State, "CRBN", "Random", cp1, new_k1, path+"crbn" + str(K) + "rand.png")
        red.plot(State, "DGARBN", "Random", cp2, new_k2, path+"dgarbn" + str(K) + "rand.png")
        red.plot(State, "DGARBN", "Random-Outdegree", cp3, new_k3, path+"dgarbn" + str(K) + "randOut.png")
        red.plot(State, "DGARBN", "Random-Ceil", cp4, new_k4, path+"dgarbn" + str(K) + "randCeil.png")
        red.plot(State, "CRBN", "Configuration Model", cp5, new_k5, path+"crbn" + str(K) + "Config.png")
        red.plot(State, "DGARBN", "Configuration Model", cp6, new_k6, path+"dgarbn"+str(K)+"Config.png")
        red.plot(State, "DGARBN", "Configuration Model-Outdegree", cp7, new_k7, path+"dgarbn"+str(K)+"ConfigOut.png")
        red.plot(State, "DGARBN", "Configuration Model-Ceil", cp8, new_k8, path+"dgarbn"+str(K)+"ConfigCeil.png")

    a1, a2, a3, a4, a5, a6, a7, a8=0, 0, 0, 0, 0, 0, 0, 0
    att1, att2, att3, att4, att5, att6, att7, att8=[0], [0], [0], [0], [0], [0], [0], [0]
    
    return ((len(att1),a1,new_k1,cp1), (len(att2),a2,new_k2,cp2), (len(att3),a3,new_k3,cp3), (len(att4),a4,new_k4,cp4),
            (len(att5),a5,new_k5,cp5), (len(att6),a6,new_k6,cp6), (len(att7),a7,new_k7,cp7), (len(att8),a8,new_k8,cp8))



    
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    start_time = time.time()
    
    K=10.0
    N=100
    p=0.5
    T=100
    P = 5
    Q = 4
    
#    X=20
    
#    initial = []
    C = []
    red=RBN(N, float(K), p)
#    red.set(K, N, T, p, P, Q, Con, Bool, periods, transitions)
#    red.CreateNetCRBN("Random")
#    red.CreateNetCRBN("Complex", "Zipf", 2.0)
#    red.CreateNetDGARBN("Random", P, Q, "outdegree")
#    red.CreateNetCRBN("Complex", distribution="Poisson", parameter=float(K))
    red.CreateNetDGARBN("Complex", P, Q, period_type="outdegree", distribution="Exponential", parameter=float(K))
    State = red.RunNetDGARBN(T)
    C.append(np.mean(red.complexity(State[-T:])))
#    print(np.count_nonzero(red.Con))
#    print((np.count_nonzero(red.Con))/N)
#    print("Periodos: ")
#    print(red.periods)
#    print("Transiciones: ")
#    print(red.transitions)
#    print("Con")
#    print(red.Con)
#    print("Bool")
#    print(red.Bool)

#    initial=np.random.randint(0, 2, N)
#    print(initial)

#    att = red.Attractors("Complex", T)
#    print(att)
#    print(len(att))
#    print( red.MeanAttractors(att) )
    
#    State=red.RunNetCRBN(2*T, initial)
#    plt.imshow(State, cmap='Greys', interpolation='None')
#    plt.xlabel('Node')
#    plt.ylabel('Time')
#    plt.title("Without perturbations")
#    plt.show()
    
#    C0=red.complexity(State[-T:])
#    print(C0)
    
#    State=red.RunNetCRBN(2*T, initial, X=X, O=1)
#    plt.imshow(State, cmap='Greys', interpolation='None')
#    plt.xlabel('Node')
#    plt.ylabel('Time')
#    plt.title("With perturbations")
#    plt.show()
    
#    C=red.complexity(State[-T:])    
#    print(C)
    
#    print(red.fragility(C,C0,X,1,N,T))
    
#    print(red.actual_k)
    red.plot(State=State,model="HeSHeTHeF",distribution="Exp-Out-Tri",complexi=C,real_K=float(K),path="/home/amahury/HeSHeTHeF.pdf")
#    red.plot("DGARBN", "Configuration Model (Zipf, parameter=2.3)", cp, "/Users/fer/Documents/doctorado3/src/graphs-rbn/dgarbnConfig.png")


    print("--- %s seconds ---" % (time.time() - start_time))
