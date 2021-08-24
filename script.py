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
    # split data at master
    df=[4,4,4,4,4]
    dflist=[]
    for i in df:
        dflist.append(i)    
    print("I'm master, I'm distributing data")
else:
    dflist=None
    
# distribute data. Note that the rank0 node is both the master and slave. Rand0 plays different roles in different time intervals.    
data = comm.scatter(dflist,root=0)
print("My rank:",rank, "My data:", data,"Time:", time.time()-start)
comm.barrier() 
###############################################################





X=np.array(rank)+data
for i in range(10):
    
    # test if barrier works one can remove this
    if rank == 1: 
        time.sleep(0.1)
    if rank == 2: 
        time.sleep(0.2)
    if rank == 3: 
        time.sleep(0.3)
    
    comm.barrier() 
    
    # sent data to master
    data1 = comm.gather(X, root=0)
    
    comm.barrier()
    
    print(data1)
    if rank == 0:
        print("rank = %s " %rank + "...receiving data to other process")
        # processing the received data at the master (which is a list)
        X=sum(data1)/5
    else:
        X=None
        
    # broadcast the processed data to all nodes    
    X=comm.bcast(X, root=0)
    X=rank+X
    print("epoch:",i,"My rank:",rank,"data",X,"Time:", time.time()-start)    
