import numpy as np
def revised_simplex(c,A,b):
    m,n=A.shape
    basic_variables=np.arange(0,m)
    nonbasic_variables=np.arange(m,n)
    initial_soln=np.zeros(n)
    initial_soln[basic_variables]=np.linalg.solve(A[:,basic_variables],b)
    while True:
        A_b=A[:,basic_variables]
        yT=np.linalg.solve(A_b.T,c[basic_variables])
        temp=True
        k=0
        for i in nonbasic_variables:
            if np.dot(yT,A[:,i])>c[i]:
                temp=False
                k=i
                break
        if temp:
            break
        else:
            db=np.linalg.solve(A_b,-A[:,k])
            d=np.zeros(n)
            d[basic_variables]=db
            d[nonbasic_variables]=np.zeros(n-m)
            d[k]=1
            if np.all(d>=0):
                return "unbounded"
            else:
                lambd=np.min(-initial_soln[d<0]/d[d<0])
                j=np.argmin(-initial_soln[d<0]/d[d<0])
                initial_soln=initial_soln+lambd*d
                nonbasic_variables=np.append(nonbasic_variables,basic_variables[j])
                nonbasic_variables=np.delete(nonbasic_variables,np.where(nonbasic_variables==k))
                nonbasic_variables=np.sort(nonbasic_variables)
                basic_variables[j]=k
                basic_variables=np.sort(basic_variables)
    print(initial_soln)
    return initial_soln

if __name__ == "__main__":
    A=np.array([[6,8,-1,0],[7,12,0,-1]])
    b=np.array([100,120])
    c=np.array([12,20,0,0])
    x=revised_simplex(c,A,b)
    if type(x)==str:
        print("The problem is unbounded")
    else:
        print("The Optimal Solution is: ",np.dot(c,x))
