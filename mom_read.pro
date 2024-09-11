;; PROCEDURE FOR READING MPI-PARALLELIZED SIMULATION DATA 
pro pic_mom_read_2d, data,nproc,nproc_i,nproc_j,pat,swap_endian=swap_endian,double=double,silent=silent

;;CHECK PROC NUM
if(nproc ne (nproc_i*nproc_j))then begin
   print,'Number of Proc. mismatch',nproc, nproc_i, nproc_j
   retall
endif

flist = file_search(pat,count=count)
if(count eq 0) then begin
   print,'No such files'
   retall
endif

;;CHECK FILE NUM CONSISTENCY WITH PROC NUM
if((count mod nproc) ne 0)then begin
   print,'parameter mismatch',nproc, count
   retall
endif
nlist = count/nproc

for ilist=0L,count-1,nproc do begin

for irank=0L,nproc-1 do begin 

   rank = 0
   for j=0L,nproc_j-1 do begin
      for i=0L,nproc_i-1 do begin
         if(irank eq rank)then begin
            irank_i = i
            irank_j = j
         endif
         rank = rank+1
      endfor
   endfor
   
   openr, /get_lun, /f77_unformatted, swap_endian=swap_endian, $
          unit, flist[irank+ilist] ;; READ DATA F77 UNFORMATTED
   if(not(keyword_set(silent)))then begin
      print,'rank=',strcompress(irank,/remove)
      print, ilist/nproc,' Reading......  ',flist[irank+ilist] 
   endif
   tmp = file_read(flist[irank+ilist].substring(0,6)+'_domain_rank='+strcompress(string(irank,format='(i04)'),/remove)+'.dat',/silent)
   nxg = long(tmp[0])
   nyg = long(tmp[1])
   nxs = long(tmp[2])
   nxe = long(tmp[3])
   nys = long(tmp[4])
   nye = long(tmp[5])

   if(irank eq 0L and ilist eq 0L)then begin
      if(keyword_set(double))then begin
         data = dblarr(nxg,nyg,6,nlist)
      endif else begin
         data = fltarr(nxg,nyg,6,nlist)
      endelse
   endif
   if(keyword_set(double))then begin
      tmp = dblarr(nxe-nxs+1,nye-nys+1,6)
   endif else begin
      tmp = fltarr(nxe-nxs+1,nye-nys+1,6)
   endelse
   readu, unit, tmp
   data[nxs:nxe,nys:nye,*,ilist/nproc] = tmp
   close,unit
   free_lun,unit

endfor
endfor

end

pro pic_mom_read, param1,param2,param3,param4,param5,swap_endian=swap_endian,double=double,silent=silent

nparam = n_params()
if(nparam eq 5)then  pic_mom_read_2d, param1, param2, param3, param4, param5, swap_endian=swap_endian, double=double, silent=silent

end