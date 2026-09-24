! Lowest-order first-family Nedelec tetrahedra, scattered-field Maxwell system.
! Bilinear (unconjugated) element matrices: real basis, complex PML tensors.
subroutine fem_assemble(nt,corners,gradients,volumes,signs,indices,n0,incident, &
                        center,halfwidth,pml_width,strength,matrices,rhs)
  use reference_math
  implicit none
  integer, intent(in) :: nt,signs(6,nt)
  real(dp), intent(in) :: corners(3,4,nt),gradients(3,4,nt),volumes(nt),indices(nt),n0
  real(dp), intent(in) :: center(3),halfwidth(3),pml_width,strength
  complex(dp), intent(in) :: incident(3,4,nt)
  complex(dp), intent(out) :: matrices(6,6,nt),rhs(6,nt)
  integer, parameter :: ends(2,6)=reshape([1,2,1,3,1,4,2,3,2,4,3,4],[2,6])
  real(dp), parameter :: qa=0.5854101966249685_dp,qb=0.1381966011250105_dp
  real(dp) :: bary(4),basis(3,6),curls(3,6),x(3),depth(3)
  complex(dp) :: stretch(3),epsilon(3),mu_inverse(3)
  integer :: t,q,a,b,i,j
  matrices=0; rhs=0
  do t=1,nt
    do a=1,6
      i=ends(1,a); j=ends(2,a)
      curls(:,a)=2*signs(a,t)*cross(gradients(:,i,t),gradients(:,j,t))
    end do
    do q=1,4
      bary=qb; bary(q)=qa
      x=matmul(corners(:,:,t),bary)
      depth=max(0.0_dp,(abs(x-center)-halfwidth)/pml_width)
      stretch=1+ii*strength*depth**2
      epsilon=product(stretch)/stretch**2
      mu_inverse=1/epsilon
      epsilon=indices(t)**2*epsilon
      do a=1,6
        i=ends(1,a); j=ends(2,a)
        basis(:,a)=signs(a,t)*(bary(i)*gradients(:,j,t)-bary(j)*gradients(:,i,t))
      end do
      do a=1,6
        rhs(a,t)=rhs(a,t)+volumes(t)/4*k0**2*(indices(t)**2-n0**2)* &
                 sum(basis(:,a)*incident(:,q,t))
        do b=1,6
          matrices(a,b,t)=matrices(a,b,t)+volumes(t)/4* &
             (sum(curls(:,a)*mu_inverse*curls(:,b))-k0**2*sum(basis(:,a)*epsilon*basis(:,b)))
        end do
      end do
    end do
  end do
end subroutine

subroutine fem_fields(nt,np,gradients,signs,coefficients,cells,bary,electric,magnetic)
  use reference_math
  implicit none
  integer, intent(in) :: nt,np,signs(6,nt),cells(np)
  real(dp), intent(in) :: gradients(3,4,nt),bary(4,np)
  complex(dp), intent(in) :: coefficients(6,nt)
  complex(dp), intent(out) :: electric(3,np),magnetic(3,np)
  integer, parameter :: ends(2,6)=reshape([1,2,1,3,1,4,2,3,2,4,3,4],[2,6])
  integer :: p,t,a,i,j
  real(dp) :: basis(3),curl_basis(3)
  electric=0; magnetic=0
  do p=1,np
    t=cells(p)
    do a=1,6
      i=ends(1,a); j=ends(2,a)
      basis=signs(a,t)*(bary(i,p)*gradients(:,j,t)-bary(j,p)*gradients(:,i,t))
      curl_basis=2*signs(a,t)*cross(gradients(:,i,t),gradients(:,j,t))
      electric(:,p)=electric(:,p)+coefficients(a,t)*basis
      magnetic(:,p)=magnetic(:,p)+coefficients(a,t)*curl_basis/(ii*k0)
    end do
  end do
end subroutine
