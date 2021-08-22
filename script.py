from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = time.time()


###############################################################
## Split and distribute data to all nodes
if rank == 0:
#     df = pd.read_csv("mimic_data/0.5missingdata.csv",index_col=0,chunksize=2000)
    df=[4,4,4,4,4]
    dflist=[]
    for i in df:
        dflist.append(i)    
    print("I'm master, I'm distributing data")
else:
    dflist=None
data = comm.scatter(dflist,root=0)
print("My rank:",rank, "My data:", data,"Time:", time.time()-start)
comm.barrier() 
###############################################################





X=np.array(rank)+data
for i in range(10):
    if rank == 1: 
        time.sleep(0.1)
    if rank == 2: 
        time.sleep(0.2)
    if rank == 3: 
        time.sleep(0.3)
    
    comm.barrier() 
    data1 = comm.gather(X, root=0)
    print(data1)
    if rank == 0:
        print("rank = %s " %rank + "...receiving data to other process")
        X=sum(data1)/5
    else:
        X=None
    X=comm.bcast(X, root=0)
    X=rank+X
    print("epoch:",i,"My rank:",rank,"data",X,"Time:", time.time()-start)    
#     for i in range(1, size):
#         data[i] = (i+1)**2
#         value = data[i]
#         print(" process %s receiving %s from process %s" % (rank , value , i))





