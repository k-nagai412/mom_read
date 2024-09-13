import os
import numpy as np
import glob


def pic_mom_read_2d(data, nproc, nproc_i, nproc_j, fname, swap_endian=False, double=False, silent=False):
  """
  Reads MPI-parallelized simulation data
  Args:
      data: Array to store the data (will be resized if needed)      => 返す配列を用意
      nproc: Total number of processes                               => 24
      nproc_i: Number of processes in i-dimension                    => 4
      nproc_j: Number of processes in j-dimension                    => 6
      pat: Pattern for file search                                   => 正規表現とかで全部読み込むパターン "0000100_bx_rank=*.dat"
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
 
  flist = glob.glob(fname)
  flist.sort()  
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
  for ilist in range(0, len(flist), nproc):
    for irank in range(nproc):
      #print(flist[irank + ilist])
      filename = flist[irank + ilist]   # 0000100_bx_rank=0000.dat,...
      with open(filename, 'rb') as f:   # f = 0000100_bx_rank=0000.dat
        if not silent:
          print('rank=', irank)
          print(ilist // nproc, 'Reading......', filename)
      
          #rank= 0
          #0 Reading...... 0000100_bx_rank=0000.dat 
        
        domainfname = os.path.join(flist[irank + ilist].split('_')[0] + '_domain_rank=' + str(irank).zfill(4) + '.dat')
        # Read domain information
        with open(domainfname,'rb') as domainf:   # domainf = 0000100_domain_rank=0000.dat
          tmp_b = domainf.read()                  #これはバイナリーデータ<class 'bytes'>
          
        #intリストにする
        tmp = np.frombuffer(tmp_b,dtype=np.int8)
        tmp_str = tmp_b.decode().strip()
        tmp_list = tmp_str.split()
        tmp = [int(num) for num in tmp_list]
        nxg, nyg, nxs, nxe, nys, nye = tmp
      #  tmp = [7007, 960, 0, 1166, 0, 239]     

      # Allocate data array (resize if needed)
        if irank == 0 and ilist == 0:                     #一番最初のループで全boxサイズのdata作る
            dtype = 'float64' if double else 'float32'      
            data = np.zeros((nxg, nyg), dtype=dtype)   
          # data.shape = (7007, 960)
        
    # Read data chunk    
        tmp_rank = np.fromfile(f, dtype=dtype, count=(nxe - nxs + 1) * (nye - nys + 1) )  #irankのdomainサイズ
        tmp_arr = tmp_rank.reshape((nxe - nxs + 1, nye - nys + 1))   
      
      # Store data chunk in main array
      data[nxs:nxe+1, nys:nye+1] = tmp_arr
      
        
    savedir ='../mom/'
    np.save(os.path.join(savedir+flist[irank + ilist].split('_')[0]+'_bx.npy'),data)
    #np.savetxt('mom_bx.dat',data,delimiter=' ')
        

def pic_mom_read(param1, param2, param3, param4, param5, swap_endian, double, silent):
  """
  Wrapper function for pic_mom_read_2d
  """
  nparam = len([p for p in (param1, param2, param3, param4, param5) if p is not None])
  if nparam == 5:
    pic_mom_read_2d(param1, param2, param3, param4, param5, swap_endian=swap_endian, double=double, silent=silent)

# Example usage (assuming param1 to param5 are appropriately defined arrays)

pic_mom_read("_", 24, 4, 6, "0003300_bx_rank=*.dat",False, False,False)
