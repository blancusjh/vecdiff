! Müller (second-kind) operator blocks: RWG currents tested with rotated
! Buffa-Christiansen (BC) functions on the barycentric refinement.
!
! Rows 1..ne:     <n x b, J> - <b,(K_e-K_i)J> - <b,(n_e^2 T_e - n_i^2 T_i)M>
! Rows ne+1..2ne: <n x b, M> - <b,(n_e^2 K_e-n_i^2 K_i)M>/c + <b,(n_e^2 T_e-n_i^2 T_i)J>/c
! with c=(n_e^2+n_i^2)/2 and T, K the PMCHWT operators (see bem.f90). Only the
! integral operators are assembled here; the sparse Gram <n x b, f> is added by
! the caller. The weighted hypersingular parts cancel exactly in n^2 T, which is
! why the div-div term of each BC/RWG pair is formed at identical points.
!
! Test sub-triangles lying in the source panel use the target-centred Duffy
! rule of bem_assemble; sub-triangles touching the source panel (their coarse
! apex vertex is a panel vertex) use the refinable product rule (qn, wn);
! distant pairs the cheap rule (qf, wf), all others (q, w). K vanishes on
! coplanar (self) pairs.
! Parallel over source panels: each thread accumulates the six columns of one
! panel privately and adds them atomically, so results do not depend on the
! thread count beyond roundoff ordering.
subroutine bem_assemble_muller(nt,ne,ns,nk,nq,nn,nf,corners,triangles,edge_ids,factors, &
                               sub_corners,sub_parent,sub_apex,sub_rows,sub_flux,n0,n1, &
                               q,w,qn,wn,qf,wf,far_ratio,matrix)
  use reference_math
  use iso_c_binding, only: c_loc, c_f_pointer
  implicit none
  integer, intent(in) :: nt,ne,ns,nk,nq,nn,nf,triangles(3,nt),edge_ids(3,nt)
  integer, intent(in) :: sub_parent(ns),sub_apex(ns),sub_rows(nk,ns)
  real(dp), intent(in) :: corners(3,3,nt),factors(3,nt),sub_corners(3,3,ns),sub_flux(3,nk,ns)
  real(dp), intent(in) :: n0,n1,q(nq),w(nq),qn(nn),wn(nn),qf(nf),wf(nf),far_ratio
  complex(dp), intent(out), target :: matrix(2*ne,2*ne)
  real(dp), pointer :: parts(:,:,:)
  real(dp) :: xp(3,max(nq,nn,nf)**2),yp(3,max(nq,nn,nf)**2)
  real(dp) :: wx(max(nq,nn,nf)**2),wy(max(nq,nn,nf)**2)
  real(dp) :: x(3),y(3),va(3),vb(3),jac,weight,c,fsub(3)
  real(dp) :: centre(3,nt),size(nt),scentre(3,ns),ssize(ns),sarea(ns)
  complex(dp) :: le(3,3,2),ke(3,3,2),l0,l1,k0j,k1j,dl
  complex(dp), allocatable :: cols(:,:)
  logical :: self,near
  integer :: p,s,t,a,b,d,u,v,n,j,m,r,col
  matrix=0
  c=(n0**2+n1**2)/2
  call c_f_pointer(c_loc(matrix), parts, [2,2*ne,2*ne])
  do p=1,nt
    centre(:,p)=sum(corners(:,:,p),dim=2)/3
    size(p)=0
    do a=1,3
      size(p)=max(size(p),norm2(corners(:,a,p)-centre(:,p)))
    end do
  end do
  do t=1,ns
    scentre(:,t)=sum(sub_corners(:,:,t),dim=2)/3
    ssize(t)=0
    do a=1,3
      ssize(t)=max(ssize(t),norm2(sub_corners(:,a,t)-scentre(:,t)))
    end do
    sarea(t)=norm2(cross(sub_corners(:,2,t)-sub_corners(:,1,t), &
                         sub_corners(:,3,t)-sub_corners(:,1,t)))/2
  end do
  !$omp parallel default(none) &
  !$omp shared(nt,ne,ns,nk,nq,nn,nf,corners,triangles,edge_ids,factors,sub_corners, &
  !$omp        sub_parent,sub_apex,sub_rows,sub_flux,n0,n1,q,w,qn,wn,qf,wf,far_ratio,parts, &
  !$omp        centre,size,scentre,ssize,sarea,c) &
  !$omp private(p,s,t,a,b,d,u,v,n,j,m,r,col,self,near,xp,yp,wx,wy,x,y,va,vb,jac, &
  !$omp         weight,fsub,le,ke,l0,l1,k0j,k1j,dl,cols)
  allocate(cols(2*ne,6))
  !$omp do schedule(dynamic,1)
  do s=1,nt
    cols=0
    do t=1,ns
      p=sub_parent(t)
      self=(p==s)
      near=self.or.any(sub_apex(t)==triangles(:,s))
      fsub=1/(2*sarea(t))
      if (near) then
        n=nn
        call triangle_rule(sub_corners(:,:,t),n,qn,wn,xp(:,1:n*n),wx(1:n*n))
        call triangle_rule(corners(:,:,s),n,qn,wn,yp(:,1:n*n),wy(1:n*n))
      else if (norm2(scentre(:,t)-centre(:,s)) > far_ratio*(ssize(t)+size(s))) then
        n=nf
        call triangle_rule(sub_corners(:,:,t),n,qf,wf,xp(:,1:n*n),wx(1:n*n))
        call triangle_rule(corners(:,:,s),n,qf,wf,yp(:,1:n*n),wy(1:n*n))
      else
        n=nq
        call triangle_rule(sub_corners(:,:,t),n,q,w,xp(:,1:n*n),wx(1:n*n))
        call triangle_rule(corners(:,:,s),n,q,w,yp(:,1:n*n),wy(1:n*n))
      end if
      le=0; ke=0
      do a=1,n*n
        x=xp(:,a)
        if(self) then
          ! Target-centred Duffy over the three sub-triangles of panel s at x.
          do d=1,3
            va=corners(:,d,s)-x
            vb=corners(:,mod(d,3)+1,s)-x
            jac=norm2(cross(va,vb))
            do u=1,nn
              do v=1,nn
                y=x+qn(u)*((1-qn(v))*va+qn(v)*vb)
                weight=wx(a)*jac*qn(u)*wn(u)*wn(v)
                call interaction(x,y,weight,sub_corners(:,:,t),corners(:,:,s), &
                  fsub,factors(:,s),[n0,n1],.true.,le,ke)
              end do
            end do
          end do
        else
          do d=1,n*n
            call interaction(x,yp(:,d),wx(a)*wy(d),sub_corners(:,:,t),corners(:,:,s), &
              fsub,factors(:,s),[n0,n1],.false.,le,ke)
          end do
        end if
      end do
      ! Contract the three sub-triangle half-functions with each BC function.
      do j=1,nk
        m=sub_rows(j,t)
        if(m<=0) cycle
        do b=1,3
          l0=sum(sub_flux(:,j,t)*le(:,b,1)); l1=sum(sub_flux(:,j,t)*le(:,b,2))
          k0j=sum(sub_flux(:,j,t)*ke(:,b,1)); k1j=sum(sub_flux(:,j,t)*ke(:,b,2))
          dl=n0**2*l0-n1**2*l1
          cols(m,b)=cols(m,b)-(k0j-k1j)
          cols(m,3+b)=cols(m,3+b)-dl
          cols(ne+m,b)=cols(ne+m,b)+dl/c
          cols(ne+m,3+b)=cols(ne+m,3+b)-(n0**2*k0j-n1**2*k1j)/c
        end do
      end do
    end do
    do b=1,3
      do r=1,2
        col=edge_ids(b,s)+(r-1)*ne
        do m=1,2*ne
          !$omp atomic
          parts(1,m,col)=parts(1,m,col)+real(cols(m,b+3*(r-1)),dp)
          !$omp atomic
          parts(2,m,col)=parts(2,m,col)+aimag(cols(m,b+3*(r-1)))
        end do
      end do
    end do
  end do
  !$omp end do
  deallocate(cols)
  !$omp end parallel
end subroutine
