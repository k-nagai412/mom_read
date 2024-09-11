n0 = 20 #particles / cell in the upstream region
nx = 530000 #maximum grid size 
ny = 1200   #maximum grid size 
nspecies = 2 #2 : ion/electron species

npg = int(nx*ny*5.0*n0*nspecies)
   #5.0 : a factor considering the downstream region and the overshoot
memsize = npg*6.0*2.5*8.0 / 1024.0**3 # GB
   #6.0 : 2D3V+ID
   #2.5 : approximate number of particle arrays (field array size is negligible)
   #8.0 : double precision in bytes

mem_max = 27.0 #Actual available memory size in each node is ~ 27GB (out of 32GB)
mpi_node = 4 #recommended for fugaku
nproc = int(memsize/(mem_max/mpi_node)/mpi_node+0.5)*mpi_node #4MPIx12thread per node
print('Nx: ',nx)
print('Ny: ',ny)
print('NPPC in the upstream: ',n0)

print('Total number of particles: ',npg)

print('memory usage [GB]: ',memsize)

print('required MPI processes on Fugaku: ',nproc)# 8GB/MPI

print('required compute nodes on Fugaku: ',nproc//4) # 4MPI/node

print('maximum number of particles in each MPI proc: ',npg//nproc) #corresponding to np in init.f90