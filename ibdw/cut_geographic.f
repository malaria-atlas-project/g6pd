! # def latfun(t):
! #     if t<.5:
! #         return (t*4-1)*np.pi
! #     else:
! #         return ((1-t)*4-1)*np.pi
! #         
! # def lonfun(t):
! #     if t<.25:
! #         return -28*np.pi/180.
! #     elif t < .5:
! #         return -28*np.pi/180. + (t-.25)*3.5
! #     else:
! #         return -169*np.pi/180.
! 
      SUBROUTINE hemisphere(lon,lat,index)
cf2py intent(out) index
      DOUBLE PRECISION lon,lat,pi
      INTEGER index
      PARAMETER (pi=3.141592653589793238462643d0)         
      
      if (lat.LT.0.0D0) then
          if ((lon.LT.-0.48869).AND.(lon.GT.-2.9496)) then
              index = 1
          else
              index = 0
          end if
      else
          if ((lon.LT.((lat*2.0D0/pi)*1.75/4.-0.48869))
     *        .AND.(lon.GT.-2.9496)) then
              index = 1
          else
              index = 0
          end if
      end if
      RETURN
      END
      

      SUBROUTINE cut_geographic(D,x,y,nx,ny,cmin,cmax,symm)
! First coordinate is longitude, second is latitude.
! Assumes r=1.

cf2py logical intent(optional) :: symm=0
cf2py integer intent(optional) :: cmin=0
cf2py integer intent(optional) :: cmax=-1
cf2py intent(hide) nx, ny
cf2py intent(inplace) D
cf2py threadsafe

      DOUBLE PRECISION D(nx,ny), x(nx,2), y(ny,2)
      integer nx,ny,j,i,i_hi,cmin,cmax, xh, yh
      LOGICAL symm
      DOUBLE PRECISION clat1, clat2, dlat, dlon, a, sterm, cterm
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      
      if (cmax.EQ.-1) then
          cmax = ny
      end if

      do j=cmin+1,cmax
        clat2 = dcos(y(j,2))
        CALL hemisphere(y(j,1), y(j,2), yh)
        if(symm) then
            D(j,j)=0.0D0            
            i_hi = j-1
        else 
            i_hi = nx
        endif
        
        do i=1,i_hi
            CALL hemisphere(x(i,1), x(i,2), xh)
            if (xh.EQ.yh) then
              clat1 = dcos(x(i,2))
              dlat = (x(i,2)-y(j,2))*0.5D0
              dlon = (x(i,1)-y(j,1))*0.5D0
              a=dsin(dlat)**2 + clat1*clat2*dsin(dlon)**2
              sterm = dsqrt(a)
              cterm = dsqrt(1.0D0-a)
              D(i,j) = 2.0D0*DATAN2(sterm,cterm)    
            else
              D(i,j) = infinity
            end if
!             if(symm) then                  
!                 D(j,i) = D(i,j)
!             end if
        enddo          
      enddo
      RETURN
      END