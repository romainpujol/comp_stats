############################################################################
#
# Romain Pujol, Lilian Welschinger
# Computational statistics, MVA
# Extension of the SAEM algorithm to left-censored data in nonlinear
# mixed-effects model: Application to HIV dynamics model
#
############################################################################

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

def f(t,phi):
    return np.log10(np.exp(phi[0])*np.exp(-np.exp(phi[2])*t)+np.exp(phi[1])*np.exp(-np.exp(phi[3])*t))

def viral_load(n_patient,mu=np.array([12,8,np.log(0.5),np.log(0.05)]),w=0.3*np.eye(4),sigma=0.065,table_t=np.array([1,3,7,14,28,56])):
    #returns the viral load of n_patient during the time table table_t, simulation of the data
    #also returns the real phi vector, will be useful for accuracy metrics
    t = len(table_t)
    viral_load_ = np.zeros((n_patient,t))
    phis = np.zeros((n_patient,4))
    for i in range(n_patient):
        epsilon = np.random.normal(0,sigma**2,t)
        phi_patient = mu + np.random.multivariate_normal(np.zeros(4),w)
        phis[i,:]=phi_patient
        for j in range(t):
            viral_load_[i,j]=f(table_t[j],phi_patient)+epsilon[j]
    return viral_load_,phis

def plot_viral_load(viral_load_,table_t=np.array([1,3,7,14,28,56])):
    #takes in input the output of the viral_load function and the time table, then plots a graph representing the viral load
    shape = viral_load_.shape
    n_patient = shape[0]
    time = shape[1]

    for i in range(n_patient):
        plt.scatter(table_t,viral_load_[i,:],color='blue',marker='+',s=5)

    plt.xlim(0,57)
    plt.hlines(2.6,0,57,colors='red',linestyles='--',label="LOQ")
    plt.legend()
    plt.show()

    return
 
def viral_load_to_censored(viral_load_,loq=2.6):
    #input : viral_load 
    #output : censored indexes

    index = np.ones(viral_load_.shape)
    for i in range(viral_load_.shape[0]):
        for j in range(viral_load_.shape[1]):
            if viral_load_[i,j]<loq:
                index[i,j]=0

    return index
    
#à faire, calculer loi de phi^m sachant y obs, y censuré (l'estimation m-1) et avec les paramètres theta^(m-1)
#simuler avec MH grâce à la loi trouvée (pour le calcul de l'acceptation rejet), les lois de proposition
#sont dans l'article
#ensuite calculer y cens(m) à partir de y obs, phi(m) et theta (m-1) (fin de la page 1566 et début 1567)
#ensuite update theta(m-1) en theta (m) et l'algo est bouclé
