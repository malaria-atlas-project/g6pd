      SUBROUTINE hemisphere(lon,lat,index)
      DOUBLE precision lon,lat
      INTEGER index
      
      
      

      SUBROUTINE geographic(D,x,y,nx,ny,cmin,cmax,symm)
! First coordinate is longitude, second is latitude.
! Assumes r=1.

cf2py logical intent(optional) :: symm=0
cf2py integer intent(optional) :: cmin=0
cf2py integer intent(optional) :: cmax=-1
cf2py intent(hide) nx, ny
cf2py intent(inplace) D
cf2py threadsafe

      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)
      integer nx,ny,j,i,i_hi,cmin,cmax
      LOGICAL symm
      DOUBLE PRECISION clat1, clat2, dlat, dlon, a, sterm, cterm
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      do j=cmin+1,cmax
        clat2 = dcos(y(j,2))
        if(symm) then
            D(j,j)=0.0D0            
            i_hi = j-1
        else 
            i_hi = nx
        endif
        
        do i=1,i_hi
            clat1 = dcos(x(i,2))
            dlat = (x(i,2)-y(j,2))*0.5D0
            dlon = (x(i,1)-y(j,1))*0.5D0
            a=dsin(dlat)**2 + clat1*clat2*dsin(dlon)**2
            sterm = dsqrt(a)
            cterm = dsqrt(1.0D0-a)
            D(i,j) = 2.0D0*DATAN2(sterm,cterm)    
!             if(symm) then                  
!                 D(j,i) = D(i,j)
!             end if
        enddo          
      enddo
      RETURN
      END