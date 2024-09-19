import os
import numpy as np
from functools import reduce

def create_databox(data, nproc, nproc_i, nproc_j, pat, dir, swap_endian=False, double=False, silent=False):
  """
  Reads MPI-parallelized simulation data
  Args:
      data: Array to store the data (will be resized if needed)
      nproc: Total number of processes
      nproc_i: Number of processes in i-dimension
      nproc_j: Number of processes in j-dimension
      pat: Pattern for file search
      swap_endian: Whether to swap endianness (default: False)
      double: Whether to use double precision (default: False)
      silent: Whether to suppress printing messages (default: False)
  """

  # Check number of processors consistency
  if nproc != nproc_i * nproc_j:
    if not silent:
      print('Number of Proc. mismatch:', nproc, nproc_i, nproc_j)
    return

  # Find data files
  #dir = '/work_stu/sarai/test/weibel'
  flist = [f for f in os.listdir(dir) if f.startswith(pat)]
  print(flist)
  if not flist:
    if not silent:
      print('No such files')
    return

  # Check consistency between file number and processes
  if len(flist) % nproc != 0:
    if not silent:
      print('parameter mismatch:', nproc, len(flist))
    return
  nlist = len(flist) // nproc

  # Loop through file lists and ranks
  chunk = []
  for ilist in range(0, len(flist), nproc):
    
    for irank in range(nproc):
      # Open file
      print(flist[irank + ilist].split('_')[0])
      print(flist[irank + ilist].split('_'))
      filename = os.path.join(dir, flist[irank + ilist].split('_')[0] + '_domain_rank=' + str(irank).zfill(4) + '.dat')
      with open(filename, 'rb') as f:
        if not silent:
          print('rank=', irank)
          print(ilist // nproc, 'Reading......', filename)

        # Read domain information
        tmp = [int(x) for x in f.read().decode().split()]
        #print(tmp)
        nxg, nyg, nxs, nxe, nys, nye = tmp
        chunk_pos = [nxs,nxe,nys,nye]
        chunk.append(chunk_pos)
        #print(chunk)

        # Allocate data array (resize if needed)
        if irank == 0 and ilist == 0:
          dtype = 'float64' if double else 'float32'
          data = np.zeros((nxg, nyg, 6, nlist), dtype=dtype)
          #print(data.shape)

        # Read data chunk
        tmp = np.zeros((nxe-nxs+1, nye-nys+1, 6), dtype=dtype)
        #tmp.reshape((nxe-nxs+1, nye-nys+1, 6))
        #print(tmp.shape)

        # Store data chunk in main array
        data[nxs:nxe+1, nys:nye+1,:, ilist // nproc] = tmp
        #print(data)
  return data, np.array(chunk)

from scipy.io import FortranFile
def load_data(data, chunk, nproc, fname):
    for i in range(nproc):
        filename = fname.format(i)
        fp = FortranFile(filename)
        values = fp.read_record(np.float32)
        fp.close()
        values = values.reshape([chunk[i][3]-chunk[i][2]+1,chunk[i][1]-chunk[i][0]+1]).T
        data[chunk[i][0]:chunk[i][1]+1, chunk[i][2]:chunk[i][3]+1] = values
        #data[chunk[i][2]:chunk[i][3]+1, chunk[i][0]:chunk[i][1]+1] = values
        #print(i)
        
        #print(values.shape)
    return data