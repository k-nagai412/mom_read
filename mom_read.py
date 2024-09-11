import os
from functools import reduce

def pic_mom_read_2d(data, nproc, nproc_i, nproc_j, pat, swap_endian=False, double=False, silent=False):
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
  flist = [f for f in os.listdir('.') if f.startswith(pat)]
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
  for ilist in range(0, len(flist), nproc):
    for irank in range(nproc):
      rank = 0
      irank_i, irank_j = 0, 0
      for j in range(nproc_j):
        for i in range(nproc_i):
          if rank == irank:
            irank_i = i
            irank_j = j
          rank += 1

      # Open file
      filename = os.path.join(flist[irank + ilist].split('_')[0], flist[irank + ilist].split('_')[0] + '_domain_rank=' + str(irank).zfill(4) + '.dat')
      with open(filename, 'rb') as f:
        if not silent:
          print('rank=', irank)
          print(ilist // nproc, 'Reading......', filename)

        # Read domain information
        tmp = [int(x) for x in f.read(6 * 4).decode().split()]
        nxg, nyg, nxs, nxe, nys, nye = tmp

        # Allocate data array (resize if needed)
        if irank == 0 and ilist == 0:
          dtype = 'float64' if double else 'float32'
          data = np.zeros((nxg, nyg, 6, nlist), dtype=dtype)

        # Read data chunk
        tmp = np.fromfile(f, dtype=dtype, count=(nxe - nxs + 1) * (nye - nys + 1) * 6)
        tmp.reshape((nxe - nxs + 1, nye - nys + 1, 6))

        # Store data chunk in main array
        data[nxs:nxe, nys:nye, :, ilist // nproc] = tmp

def pic_mom_read(param1, param2, param3, param4, param5, swap_endian=False, double=False, silent=False):
  """
  Wrapper function for pic_mom_read_2d
  """
  nparam = len([p for p in (param1, param2, param3, param4, param5) if p is not None])
  if nparam == 5:
    pic_mom_read_2d(param1, param2, param3, param4, param5, swap_endian=swap_endian, double=double, silent=silent)

# Example usage (assuming param1 to param5 are appropriately defined arrays)
pic_mom_read(param1, param2, param3, param4, param5, double=True)