import numpy as np
import sympy

def ipopt(c,A,b):
    if np.linalg.matrix_rank(A)<np.min(A.shape):
        A=A[list(sympy.Matrix(A).T.rref()[1])]
    m,n=A.shape
    primal=np.ones(n,dtype=np.float64)
    dual=np.ones(m,dtype=np.float64)
    slack=np.ones(n,dtype=np.float64)
    flag=(abs(np.dot(primal,slack))>1e-6)
    while flag:
        sig=0.4
        mu=np.dot(primal,slack)/n
        b2=np.zeros(m+2*n,dtype=np.float64)
        b2[:m]=b-np.dot(A,primal)
        b2[m:m+n]=c-np.dot(A.T,dual)-slack
        b2[m+n:m+2*n]=sig*mu*np.ones(n,dtype=np.float64)-np.dot(np.dot(np.diag(primal),np.diag(slack)),np.ones(n,dtype=np.float64))
        A2=np.zeros((m+2*n,m+2*n),dtype=np.float64)
        A2[:m,:n]=A
        A2[m:m+n,n:m+n]=A.T
        A2[m:m+n,n+m:m+2*n]=np.eye(n,dtype=np.float64)
        A2[m+n:m+2*n,:n]=np.diag(slack)
        A2[m+n:m+2*n,m+n:m+2*n]=np.diag(primal)
        delta=np.linalg.solve(A2,b2)
        dx,dl,ds=delta[:n],delta[n:m+n],delta[n+m:m+2*n]
        alpha_max=1
        neg_dx=np.where(dx<0)[0]
        neg_ds=np.where(ds<0)[0]
        if len(neg_dx)>0:
            alpha_max=min(alpha_max,np.min(-primal[neg_dx]/dx[neg_dx]))
        if len(neg_ds)>0:
            alpha_max=min(alpha_max,np.min(-slack[neg_ds]/ds[neg_ds]))
        alpha=min(0.99*alpha_max,1)
        primal+=alpha*dx
        dual+=alpha*dl
        slack+=alpha*ds
        flag=(abs(np.dot(primal,slack))>1e-6)
    return primal

if __name__ == "__main__":
    A=np.array([[6,8,-1,0],[7,12,0,-1]],dtype=np.float64)
    b=np.array([100,120],dtype=np.float64)
    c=np.array([12,20,0,0],dtype=np.float64)
    x=ipopt(c,A,b)
    print(x)
    print("The Optimal Solution is: ",np.dot(c,x))