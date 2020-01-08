# encoding: utf-8
# pset2.py

import numpy as np
# don't forget import packages, e.g. scipy
# but make sure you didn't put unnecessary stuff in here

# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n): # 5 pts
    n = len(diag_broadcast)
    d = np.zeros(5)
    
    L = np.zeros((2,n-1))
    U = np.zeros((n,3))   
    
    if n == 3:
        d[1:4] = diag_broadcast
    else:
        d = diag_broadcast

    
    a, b, c, e = d[2], d[3], d[1], d[2]
    for k in range(n-2):
        a1 = e- b*c/a
        b1 = d[3] - d[4]*c/a
        c1 = d[1] - d[0]*b/a
        e1 = d[2] - d[4]*d[0]/a
        
        L[0][k], L[1][k] = c/a, d[0]/a
        U[k][0], U[k][1],U[k][2] = a, b, d[4]
        a, b, c, e = a1,b1,c1,e1
    
    L[0][n-2] = c/a
    U[n-2][0], U[n-1][0] = a, e - b*c/a
    U[n-2][1] = b
    
    if n == 3:
        return L[:1], U[:, :2]
    else:
        return L,U
    return L, U


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR

def gram_schmidt_qr(A): # 5 pts
    # your code is here
    m,n = A.shape
    
    Q = np.zeros((n,n))        
    R = np.zeros_like(A)
    R[0,0] = np.linalg.norm(A[:,1],2)
    Q[:,0] = A[:,0]/R[0,0]
    for k in range(1, n):   
        R[:k-1,k] = np.dot(Q[:m,:k-1].T,A[:m,k])
        z = A[:m,k] - np.dot(Q[:m,:k-1],R[:k-1,k])
        R[k,k] = np.linalg.norm(z,2)
        Q[:m,k] = z / R[k,k]#** 2

    R = np.around(R, decimals=10)
        
    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def modified_gram_schmidt_qr(A): # 5 pts
    m,n = A.shape
    
    Q = np.zeros((n,n))        
    R = np.zeros((m,n))
 
    for k in range(0, n):   
        R[k,k] = np.linalg.norm(A[:m,k],2)
        Q[:m,k] = A[:m,k]/R[k,k]
        for j in range(k+1,n):
            R[k,j] = np.dot(Q[:m,k].T,A[:m,j])
            A[:m,j] = A[:m,j] - np.dot(Q[:m,k],R[k,j])

    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A=QR

def householder_qr(A):
    m,n = A.shape
    Q = np.eye(m)
    R = np.copy(A)
    
    for k in range(n): 
        x = np.copy(R[k:, k:k+1])
        e = np.zeros((x.size, 1))
        e[0, 0] = 1
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e
        v = v/np.linalg.norm(v)
        H_k = np.eye(n)
        H_k[k:, k:] = np.eye(n-k) - 2 * v@v.T
        Q = Q@H_k
        R = H_k@R
        
    return Q, R

# INPUT:  G - np.ndarray
# OUTPUT: A - np.ndarray (of size G.shape)
def pagerank_matrix(G): # 5 pts
    A = np.zeros_like(G, dtype='float64')
    for i in range (G.shape[0]):
        for j in range(G.shape[1]):           
            if np.sum(G[:,j]) == 0: #G[:,j] outgoing
                A[i,j] = 0
            else:
                A[i,j] = G[i,j]/np.sum(G[:,j]) 
    return A


# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive) 
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter): # 5 pts
    x = x0   
    res = np.zeros(num_iter + 1) 
    for i in range(num_iter+1):
        x = A@x
        x = x/np.linalg.norm(x,2)        
        l = (A@x).T@x
        res[i] = np.linalg.norm(A@x - l*x ,2)
        
    return x, l, res


# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x): # 2 pts
    N = A.shape[1]
    y = d*A@x + (1-d)/N*np.ones((N,N))
    return y


def return_words():
    # insert the (word, cosine_similarity) tuples
    # for the words 'numerical', 'linear', 'algebra' words from the notebook
    # into the corresponding lists below
    # words_and_cossim = [('word1', 'cossim1'), ...]
    model = WordVectors(vocab, W)
    
    numerical_words_and_cossim = model.nearest_words("numerical")
    linear_words_and_cossim = model.nearest_words("linear")
    algebra_words_and_cossim = model.nearest_words("algebra")
    
    return numerical_words_and_cossim, linear_words_and_cossim, algebra_words_and_cossim