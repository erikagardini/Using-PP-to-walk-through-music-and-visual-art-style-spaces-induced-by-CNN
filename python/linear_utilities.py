#References:
#[1] 'Finding Prinicpal Paths in Data Space', M.J.Ferrarotti, W.Rocchia, S.Decherchi
#[2] 'Design and HPC Implementation of Unsupervised Kernel Methods in the Context of Molecular Dynamics', M.J.Ferrarotti, PhD Thesis.
#[3] https://github.com/mjf-89/PrincipalPath/blob/master/principalpath.py

import numpy as np
from scipy.spatial import distance

def initMedoids(X, n, init_type, exclude_ids=[]): 
    """
    Initialize NC medoids with init_type rational.

    Args:
        [ndarray float] X: data matrix

        [int] n: number of medoids to be selected
        
        [string] init_type: rational to be used
            'uniform': randomly selected with uniform distribution
            'kpp': k-means++ algorithm

        [ndarray int] exclude_ids: blacklisted ids that shouldn't be selected

    Returns:
        [ndarray int] med_ids: indices of the medoids selected
    """

    N=X.shape[0]
    D=X.shape[1]
    med_ids=-1*np.ones(n,int)
    np.random.seed(123)

    if(init_type=='uniform'):
        while(n>0):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(med_ids==med_id)==0 and np.count_nonzero(exclude_ids==med_id)==0):
                med_ids[n-1]=med_id
                n = n-1

    elif(init_type=='kpp'):
        accepted = False
        while(not accepted):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(exclude_ids==med_id)==0):
                accepted = True
        med_ids[0]=med_id

        for i in range(1,n):
            Xmed_dst = distance.cdist(X,np.vstack([X[med_ids[0:i],:],X[exclude_ids,:]]),'sqeuclidean') 
            D2 = Xmed_dst.min(1)
            D2_n = 1.0/np.sum(D2)
            accepted = False
            while(not accepted):
                med_id = np.random.randint(0,N)
                if(np.random.rand()<D2[med_id]*D2_n):
                    accepted = True
            med_ids[i]=med_id
    else:
        raise ValueError('init_type not recognized.')

    return(med_ids)

def find_elbow(f):
    """
    Find the elbow in a function f, as the point on f with max distance from the line connecting f[0,:] and f[-1,:]

    Args:
        [ndarray float] f: function (Nx2 array in the form [x,f(x)]) 

    Returns:
        [int]  elb_id: index of the elbow 
    """
    ps = np.asarray([f[0,0],f[0,1]])
    pe = np.asarray([f[-1,0],f[-1,1]])
    p_line_dst = np.ndarray(f.shape[0]-2,float)
    for i in range(1,f.shape[0]-1):
        p = np.asarray([f[i,0],f[i,1]])

        mtx_1 = pe - ps
        mtx_2 = ps - p
        mtx_3 = pe - ps

        cp = np.cross(mtx_1,mtx_2)
        den = np.linalg.norm(cp)
        num = np.linalg.norm(mtx_3)
        p_line_dst[i-1] = den / num
    elb_id = np.argmax(p_line_dst)+1

    return elb_id
