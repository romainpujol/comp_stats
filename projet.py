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
from scipy.stats import multivariate_normal, norm
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
    
def p(phi,i,y,mu_current,omega_current,sigma_current,table_t=np.array([1,3,7,14,28,56])):
    #y represents the whole matrix (with imputed values of ycensored)
    #i represents the line we use
    N,K = y.shape
    value = 0
    if omega_current == 0:
        print("error, omega = 0")
    else:
        matrix=(1/omega_current)*np.eye(4)
        value -=(phi-mu_current).T @ matrix @ (phi-mu_current)/2
        for j in range(K):
            value -= (1/(2*sigma_current))*(y[i,j]-f(table_t[j],phi))**2

    return np.exp(value)

def proposition_prior(current_mu,current_omega,current_phi,current_sigma,current_y):
    #current_mu, current_omega are two parameters that are being constructed in the procedure
    N,K = current_phi.shape
    update = np.zeros((N,K))
    for i in range(N):
        proposition_=np.random.multivariate_normal(current_mu,current_omega*np.eye(4))
        q_ratio = multivariate_normal.pdf(proposition_,current_mu,current_omega*np.eye(4))/multivariate_normal.pdf(current_phi[i,:],current_mu,current_omega*np.eye(4))
        acc_rate = np.exp(np.minimum(0,np.log(q_ratio)+np.log(p(proposition_,i,current_y,current_mu,current_omega,current_sigma))-np.log(p(current_phi[i,:],i,current_y,current_mu,current_omega,current_sigma))))
        u = np.random.rand()
        if u<acc_rate:
            update[i,:]=proposition_
        else:
            update[i,:]=current_phi[i,:]
    return update

def proposition_multidim_rw(current_phi,current_omega,current_sigma,current_y,current_mu,lambd=1):
    #multidimensional random walk proposal
    N,K = current_phi.shape
    proposition = np.zeros((N,K))
    for i in range(N):
        proposition_=np.random.multivariate_normal(current_phi[i,:],lambd*current_omega*np.eye(4))
        acc_rate = np.exp(np.minimum(0,np.log(p(proposition_,i,current_y,current_mu,current_omega,current_sigma))-np.log(p(current_phi[i,:],i,current_y,current_mu,current_omega,current_sigma))))
        u = np.random.rand()
        if u<acc_rate:
            proposition[i,:]=proposition_
        else:
            proposition[i,:]=current_phi[i,:]
    return proposition

def proposition_unidim_rw(current_phi,lambd,current_y,current_mu,current_omega,current_sigma):
    #unidimensional random walk proposal
    N,K = current_phi.shape
    proposition = np.copy(current_phi)
    for i in range(N):
        line_i = np.copy(current_phi[i,:])
        for j in range(K):
            step = np.random.normal(0,lambd)
            proposition[i,j]+=step
            acc_rate = np.minimum(1,p(proposition[i,:],i,current_y,current_mu,current_omega,current_sigma)/p(line_i,i,current_y,current_mu,current_omega,current_sigma))
            u = np.random.rand()
            if u>acc_rate:
                proposition[i,j]=line_i[j]
    
    return proposition


def hm_algorithm_update(current_y,current_phi,current_omega,current_mu,current_sigma,lambd,method):
    if method == 1:
        update= proposition_prior(current_mu,current_omega,current_phi,current_sigma,current_y)
    elif method ==2:
        update=proposition_multidim_rw(current_phi,current_omega,current_sigma,current_y,current_mu,lambd=1)
    elif method ==3:
        update=proposition_unidim_rw(current_phi,lambd,current_y,current_mu,current_omega,current_sigma)
    return update    
    


def update_y_cens(phi,t,sigma,LOQ):
    m=f(t,phi) 
    C=(f(t,phi)-LOQ)/sigma
    alpha = (C+np.sqrt(C**2+4))/2
    u = np.random.rand()
    x = -1/alpha*np.log(1-u)+C
    proba = np.exp(-(x-alpha)**2/2)
    u = np.random.rand()
    while u > proba :
        u = np.random.rand()
        x = -1/alpha * np.log(1 - u) + C
        proba = np.exp(-(x-alpha)**2/2)
        u = np.random.rand()
    y = m - sigma*x
    return y


mu=np.array([12,8,np.log(0.5),np.log(0.05)])
w=0.3*np.eye(4)
sigma=0.065
table_t=np.array([1,3,7,14,28,56])

y,phi = viral_load(40)
phi_test = phi +0.5
print("real value")
print(phi[0:3,:])
print("avant update")
print(phi_test[0:3,:])
print("après update")
print(hm_algorithm_update(y,phi_test,0.3,mu,sigma,0.3,3)[0:3,:])

#à faire, calculer loi de phi^m sachant y obs, y censuré (l'estimation m-1) et avec les paramètres theta^(m-1)
#simuler avec MH grâce à la loi trouvée (pour le calcul de l'acceptation rejet), les lois de proposition
#sont dans l'article
#ensuite calculer y cens(m) à partir de y obs, phi(m) et theta (m-1) (fin de la page 1566 et début 1567)
#ensuite update theta(m-1) en theta (m) et l'algo est bouclé


#travailler sur l'acceptance rate de proposition_multidim_rw et proposition_unidim_rw
