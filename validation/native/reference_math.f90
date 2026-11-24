! Numerical conventions: exp(-i omega t); coordinates in vacuum wavelengths.
module reference_math
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: dp = real64
  real(dp), parameter :: pi = acos(-1.0_dp), k0 = 2*pi
  complex(dp), parameter :: ii = (0.0_dp, 1.0_dp)
contains
  pure function cross(a,b) result(c)
    real(dp), intent(in) :: a(3),b(3)
    real(dp) :: c(3)
    c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)]
  end function

  subroutine triangle_rule(v,n,q,w,points,weights)
    integer, intent(in) :: n
    real(dp), intent(in) :: v(3,3),q(n),w(n)
    real(dp), intent(out) :: points(3,n*n),weights(n*n)
    real(dp) :: jac
    integer :: a,b,j
    jac = norm2(cross(v(:,2)-v(:,1),v(:,3)-v(:,1)))
    j=0
    do a=1,n
      do b=1,n
        j=j+1
        points(:,j)=v(:,1)+q(a)*(v(:,2)-v(:,1))+(1-q(a))*q(b)*(v(:,3)-v(:,1))
        weights(j)=jac*(1-q(a))*w(a)*w(b)
      end do
    end do
  end subroutine

  subroutine interaction(x,y,weight,vx,vy,fx,fy,indices,omit_curl,le,ke)
    real(dp), intent(in) :: x(3),y(3),weight,vx(3,3),vy(3,3),fx(3),fy(3),indices(2)
    logical, intent(in) :: omit_curl
    complex(dp), intent(inout) :: le(3,3,2),ke(3,3,2)
    real(dp) :: distance,d(3),fi(3),fj(3),wave
    complex(dp) :: green,radial
    integer :: a,b,m
    d=x-y
    distance=norm2(d)
    do m=1,2
      wave=k0*indices(m)
      green=weight*exp(ii*wave*distance)/(4*pi*distance)
      radial=green*(ii*wave-1/distance)/distance
      do a=1,3
        fi=fx(a)*(x-vx(:,a))
        do b=1,3
          fj=fy(b)*(y-vy(:,b))
          le(a,b,m)=le(a,b,m)+ii*k0*green*(dot_product(fi,fj)-4*fx(a)*fy(b)/wave**2)
          if (.not.omit_curl) ke(a,b,m)=ke(a,b,m)+radial*dot_product(fi,cross(d,fj))
        end do
      end do
    end do
  end subroutine
end module
