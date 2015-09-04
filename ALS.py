# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:02:54 2015

Collaborative filtering Alternate Least Squares optimization algorithm

@author: Xiaonan Lu
"""
import numpy as np
import math as math
import matplotlib.pyplot as plt
#from numpy.linalg import inv

def getRMSE(R, U, M, W):
    """Return RMSE
    Args:
        R: the user-item matrix
        U: User matrix
        M: Item matrix
        W: indicator matrix of R
    """
    print ('begin calculating RMSE')
    SquareError = 0
    for i, Ui in enumerate(U.transpose()):
        SquareError += (np.linalg.norm((R[i,:] - np.dot(Ui, M))*W[i,:])**2)
    return(math.sqrt(SquareError/W.sum()))
    
def runALS(R, lambdaa, n_features, n_bp):
    """Run the Alternate Least Squares optimization. Return R_hat and RMSE per iteration
    Args:
        R: the user-item matrix
        lambdaa: the regularization paramter
        n_features: number of latent factors for user and item
        n_bp: the iteration stopping criteria
    """
    W = R > 0

    # initialize variables
    n_users, n_movies = R.shape
    U = np.random.rand(n_features, n_users)
    M = np.random.rand(n_features, n_movies)
    errors = []
    error_dec = 1
    
    while error_dec > n_bp :
        print('start a new iteration, previous error_dec is:', error_dec)
        for i, Ui in enumerate(U.transpose()):
            #optimize the user matrix
            n_Ui = W[i,:].sum()
            Ii = W[i,:] == 1
            M_Ii = M[:,Ii]
            R_Ii = R[i,Ii]
            Ai = np.dot(M_Ii, M_Ii.transpose()) + lambdaa*n_Ui*np.identity(n_features)
            Vi = np.dot(M_Ii, R_Ii.transpose())
            U[:,i] = np.linalg.solve(Ai, Vi)
            #U[:,i] = np.dot(inv(Ai), Vi)

        for j, Mj in enumerate(M.transpose()):
            #optimize the movie matrix
            n_Mj = W[:,j].sum()
            Ij = W[:,j] == 1
            U_Ij = U[:, Ij.transpose()]
            R_Ij = R[Ij, j]
            Aj = np.dot(U_Ij, U_Ij.transpose()) + lambdaa*n_Mj*np.identity(n_features)
            Vj = np.dot(U_Ij,R_Ij)
            M[:,j] = np.linalg.solve(Aj, Vj) 
            #M[:,j] = np.dot(inv(Aj), Vj)
            
        errors.append(getRMSE(R,U,M,W))
        if (len(errors) >= 2):
            error_dec = errors[-2] - errors[-1]
             
    plt.plot(errors)
    plt.xlabel('iteration number')
    plt.ylabel('RMSE')

    print (errors)
    
    return (U, M)
    