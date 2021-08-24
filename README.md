# mpi4py_distributed_master_slave
This repo gives an easy example of master/slave framework for distributed computing.

Please install the mpi4py by the following command with Anaconda.
1. enter your env
> conda activate myenv
2. install mpi4py
> conda install mpi4py

Run
> mpiexec -np 5 python script.py

To compare time properly, (e.g. while using sklearn, it will automatical occupy all the resources.), one has to set the restriction by the following commands.

Run
> OMP_NUM_THREADS=1 mpiexec -np 5 python script.py
