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
#np.random.seed(1)

def f(t,phi):
    return np.log10(np.exp(phi[0])*np.exp(-np.exp(phi[2])*t)+np.exp(phi[1])*np.exp(-np.exp(phi[3])*t))

def viral_load(n_patient,mu=np.array([12,8,np.log(0.5),np.log(0.05)]),w=0.3*np.ones(4),sigma=0.065,table_t=np.array([1,3,7,14,28,56])):
    #returns the viral load of n_patient during the time table table_t, simulation of the data
    #also returns the real phi vector, will be useful for accuracy metrics
    t = len(table_t)
    viral_load_ = np.zeros((n_patient,t))
    phis = np.zeros((n_patient,4))
    for i in range(n_patient):
        epsilon = np.random.normal(0,np.sqrt(sigma),t)
        phi_patient = mu + np.random.multivariate_normal(np.zeros(4),np.diag(w))
        phis[i,:]=phi_patient
        for j in range(t):
            viral_load_[i,j]=f(table_t[j],phi_patient)+epsilon[j]
    return viral_load_,phis

def plot_viral_load(viral_load_,real_mu,approx_mu,table_t=np.array([1,3,7,14,28,56])):
    #takes in input the output of the viral_load function and the time table, then plots a graph representing the viral load
    shape = viral_load_.shape
    n_patient = shape[0]
    time = shape[1]

    for i in range(n_patient):
        plt.scatter(table_t,viral_load_[i,:],color='blue',marker='+',s=5)

    trace = np.log10(np.exp(real_mu[0])*np.exp(-0.5*table_t)+np.exp(real_mu[1])*np.exp(-0.05*table_t))
    plt.plot(table_t,trace,label="Real trend")
    trace_app = np.log10(np.exp(approx_mu[0])*np.exp(-np.exp(approx_mu[2])*table_t)+np.exp(approx_mu[1])*np.exp(-np.exp(approx_mu[3])*table_t))
    plt.plot(table_t,trace_app,label="Approximate trend")
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
    #if omega_current == 0:
    #    print("error, omega = 0")
    #else:
    matrix = np.diag(1/omega_current)
    value =-(phi-mu_current).T @ matrix @ (phi-mu_current)/2
    for j in range(K):
        value -= (1/(2*sigma_current))*((y[i,j]-f(table_t[j],phi))**2)

    return np.exp(value)

def proposition_prior(current_mu,current_omega,current_phi,current_sigma,current_y):
    #current_mu, current_omega are two parameters that are being constructed in the procedure
    N,K = current_phi.shape
    update = np.zeros((N,K))
    for i in range(N):
        proposition_=np.random.multivariate_normal(current_mu,np.diag(current_omega))
        q_ratio = multivariate_normal.pdf(proposition_,current_mu,np.diag(current_omega))/multivariate_normal.pdf(current_phi[i,:],current_mu,np.diag(current_omega))
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
        proposition_=np.random.multivariate_normal(current_phi[i,:],lambd*np.diag(current_omega))
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
    if method==1:
        update= proposition_prior(current_mu,current_omega,current_phi,current_sigma,current_y)
    elif method==2:
        update=proposition_multidim_rw(current_phi,current_omega,current_sigma,current_y,current_mu,lambd=1)
    elif method==3:
        update=proposition_unidim_rw(current_phi,lambd,current_y,current_mu,current_omega,current_sigma)
    return update    
    


def update_y_cens(phi,t,sigma,LOQ):
    m=f(t,phi) 
    C=(f(t,phi)-LOQ)/sigma
    alpha = (C+np.sqrt(C**2+4))/2
    u = np.random.rand()
    x = (-1/alpha)*np.log(1-u)+C
    proba = np.exp(-(x-alpha)**2/2)
    u = np.random.rand()
    while u > proba :
        u = np.random.rand()
        x = -1/alpha * np.log(1 - u) + C
        proba = np.exp(-(x-alpha)**2/2)
        u = np.random.rand()
    y = m - sigma*x
    return y


def S(y,phi,time_t=np.array([1,3,7,14,28,56])):
    N,K = y.shape
    S_1 = np.sum(phi,axis=0)
    S_2 = np.sum(phi**2,axis=0)
    S_3=0
    for i in range(N):
        for j in range(K):
            S_3+=((y[i,j]-f(time_t[j],phi[i,:]))**2)
        
    return S_1,S_2,S_3

def update_s(s_m,gamma,y,phi):
    s_1,s_2,s_3 = S(y,phi)
    if gamma == 1:
        return s_1,s_2,s_3
    else:
        return s_m[0] +gamma*(s_1-s_m[0]),s_m[1] +gamma*(s_2-s_m[1]),s_m[2] +gamma*(s_3-s_m[2])


def update_mu_w_sigma(S_1,S_2,S_3,N,K):
    new_mu = S_1/N
    new_w = S_2/N- (new_mu)**2
    new_sigma = S_3/(N*K)
    return new_mu,new_w,new_sigma

def SAEM(omega_0,mu_0,sigma_0,y_0,phi_0,s_0,iteration,method,lambd=1,t=np.array([1,3,7,14,28,56]),LOQ=2.6,M1 = 1000):

    y = np.copy(y_0)
    phi = np.copy(phi_0)
    omega = np.copy(omega_0)
    mu = np.copy(mu_0)
    sigma = np.copy(sigma_0)
    s = s_0
    N,K = y.shape
    mu_evolve = [mu]
    omega_evolve = [omega]
    sigma_evolve = [sigma]

    index_censored = np.where(y <= LOQ)

    for i in range(iteration):
        #########################################
        # S step : simulation of the missing data
        #########################################

        #simulate phi
        phi = hm_algorithm_update(y,phi,omega,mu,sigma,lambd,method)
        #simulate y_censored
        for k,j in zip(index_censored[0],index_censored[1]):
            y[k,j] = update_y_cens(phi[k,:],t[j],sigma,LOQ)

        #########################################
        # SA step 
        #########################################

        #update s
        if i <= M1:
            gamma = 1
        else:
            gamma = 1/(i-M1)
        
        s = update_s(s,gamma,y,phi)

        #########################################
        # M step 
        #########################################

        mu,omega,sigma = update_mu_w_sigma(s[0],s[1],s[2],N,K)
        mu_evolve.append(mu)
        sigma_evolve.append(sigma)
        omega_evolve.append(omega)

    return mu,omega,sigma,y,mu_evolve,sigma_evolve,omega_evolve





################################################################
# Real parameters and data
################################################################

mu_=np.array([12,8,np.log(0.5),np.log(0.05)])
w=0.3*np.ones(4)
sigma=0.065
table_t=np.array([1,3,7,14,28,56])
y,phi = viral_load(100)


################################################################
# Input of SAEM, censored data
################################################################

starting_y=np.copy(y)
N,K = starting_y.shape
for i in range(N):
    for j in range(K):
        if starting_y[i,j]<2.6:
            ##############################################################
            # ici on peut expérimetner avec la valeur à mettre au départ
            #par exemple 2.6, 1.3, 0
            #ou d'autres idées un peu plus intéressante en calculant
            #l'écart avec 2.6 avec la valeur précédente,
            #on peut peut être imputer une valeur intelligente :
            # si la valeur d'avant est 2.7, on peut se douter que celle
            # d'après sera vraisemblablement plus faible que si la valeur
            #précédente était 3.6
            ##############################################################
            starting_y[i,j]=2.6 



#######################################################################
# ici on peut aussi essayer de donner des valeurs intéressantes
# pour certaines variables, j'y ai pas pensé encore
#j'ai juste mis des trucs aléatoires proche de la vraie valeur
#######################################################################

phi_test = phi + 2*(np.random.rand(4))-1
s_0 = (np.zeros(4), np.zeros(4), 0)
starting_mu=mu_+4*(np.random.rand(4))-2
starting_w= w +0.2*(np.random.rand(4))-0.1
starting_sigma = 0.4


#problème avec la méthode 1 dans la partie d'exploration : quand on est en dessous de M1 parce qu'on passe sur une matrice
#qui n'est plus SDP en faisant varier les valeurs de theta (donc les valeurs de omega), 
#en revanche, les méthodes 2 et 3 fonctionnent plutôt bien, je n'ai pas vu de problème

method = 2
mu,omega,sigma,y,mu_evolve,sigma_evolve,omega_evolve=SAEM(starting_w,starting_mu,starting_sigma,starting_y,phi_test,s_0,1500,method,M1=500)

#alors oui des valeurs ne convergent pas, notamment celles de w mais dans l'article c'est pareil
#et les mecs de l'année précédente aussi (et ils convergent moins bien pour les autres valeurs)

"""

montre les paramètres et la tendance
plot_viral_load(y,mu_,mu)

"""

"""
graphique des paramèters
"""


mu_evolve=np.array(mu_evolve)
omega_evolve = np.array(omega_evolve)
sigma_evolve = np.array(sigma_evolve)

fig, axs = plt.subplots(3, 4, figsize=(10, 8))

axs[0, 0].plot(mu_evolve[:,0], label='ln(P1)', color='blue')
axs[0, 0].axhline(12, color='r', linestyle='--')  
axs[0, 0].set_ylim(10,14)
axs[0, 0].legend()

axs[0, 1].plot(mu_evolve[:,1], label='ln(P2)', color='blue')
axs[0, 1].axhline(8, color='r', linestyle='--')  
axs[0, 1].set_ylim(6,10)
axs[0, 1].legend()

axs[0, 2].plot(mu_evolve[:,2], label='ln(lambda1)', color='blue')
axs[0, 2].axhline(np.log(0.5), color='r', linestyle='--')  
axs[0, 2].set_ylim(np.log(0.5)-2,np.log(0.5)+2)
axs[0, 2].legend()

axs[0, 3].plot(mu_evolve[:,3], label='ln(lambda2)', color='blue')
axs[0, 3].axhline(np.log(0.05), color='r', linestyle='--')  
axs[0, 3].set_ylim(np.log(0.05)-2,np.log(0.05)+2)
axs[0, 3].legend()

axs[1, 0].plot(omega_evolve[:,0], label='w_1', color='blue')
axs[1, 0].axhline(0.3, color='r', linestyle='--')  
axs[1, 0].set_ylim(0,1)  
axs[1, 0].legend()

axs[1, 1].plot(omega_evolve[:,1], label='w_2', color='blue')
axs[1, 1].axhline(0.3, color='r', linestyle='--')  
axs[1, 1].set_ylim(0,1)  
axs[1, 1].legend()

axs[1, 2].plot(omega_evolve[:,2], label='w_3', color='blue')
axs[1, 2].axhline(0.3, color='r', linestyle='--')
axs[1, 2].set_ylim(0,1)
axs[1, 2].legend()

axs[1, 3].plot(omega_evolve[:,3], label='w_4', color='blue')
axs[1, 3].axhline(0.3, color='r', linestyle='--')
axs[1, 3].set_ylim(0,1)  
axs[1, 3].legend()

axs[2, 0].plot(sigma_evolve, label='sigma', color='blue')
axs[2, 0].axhline(0.065, color='r', linestyle='--')
axs[1, 0].set_ylim(0,0.5)  
axs[2, 0].legend()

axs[2, 1].axis('off')
axs[2, 2].axis('off')
axs[2, 3].axis('off')

plt.tight_layout()
plt.show()



#travailler sur l'acceptance rate de proposition_multidim_rw et proposition_unidim_rw avec le lambda
#sur la première imputation
