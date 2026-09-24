! PMCHWT weak formulation with continuous RWG surface currents.
! Self panels use a target-centred Duffy transform; adjacent panels use a
! separately refinable product quadrature. No source softening or fitted currents.
! Pairs of panels farther apart than far_ratio times their summed sizes (centroid
! to farthest corner) use the cheaper rule (qf, wf); the kernel is smooth there.
! Assembly and field evaluation are OpenMP parallel; assembly accumulates with
! atomic updates on a real view of the matrix, so the result does not depend on
! the thread count beyond roundoff ordering.
subroutine bem_assemble(nt,ne,nq,nn,nf,corners,triangles,edge_ids,factors,n0,n1, &
                        q,w,qn,wn,qf,wf,far_ratio,matrix)
  use reference_math
  use iso_c_binding, only: c_loc, c_f_pointer
  implicit none
  integer, intent(in) :: nt,ne,nq,nn,nf,triangles(3,nt),edge_ids(3,nt)
  real(dp), intent(in) :: corners(3,3,nt),factors(3,nt),n0,n1,q(nq),w(nq),qn(nn),wn(nn)
  real(dp), intent(in) :: qf(nf),wf(nf),far_ratio
  complex(dp), intent(out), target :: matrix(2*ne,2*ne)
  real(dp), pointer :: parts(:,:,:)
  real(dp) :: xp(3,max(nq,nn,nf)**2),yp(3,max(nq,nn,nf)**2)
  real(dp) :: wx(max(nq,nn,nf)**2),wy(max(nq,nn,nf)**2)
  real(dp) :: x(3),y(3),va(3),vb(3),jac,weight,centre(3,nt),size(nt)
  complex(dp) :: le(3,3,2),ke(3,3,2),values(4)
  logical :: near
  integer :: p,s,a,b,c,d,u,v,n,i,j,k,rows(4),cols(4)
  matrix=0
  call c_f_pointer(c_loc(matrix), parts, [2,2*ne,2*ne])
  do p=1,nt
    centre(:,p)=sum(corners(:,:,p),dim=2)/3
    size(p)=0
    do a=1,3
      size(p)=max(size(p),norm2(corners(:,a,p)-centre(:,p)))
    end do
  end do
  !$omp parallel do schedule(dynamic,4) default(none) &
  !$omp shared(nt,ne,nq,nn,nf,corners,triangles,edge_ids,factors,n0,n1,q,w,qn,wn, &
  !$omp        qf,wf,far_ratio,parts,centre,size) &
  !$omp private(p,s,a,b,c,d,u,v,n,i,j,k,near,xp,yp,wx,wy,x,y,va,vb,jac,weight, &
  !$omp         le,ke,values,rows,cols)
  do p=1,nt
    do s=p,nt
      near=.false.
      do a=1,3
        if(any(triangles(a,p)==triangles(:,s))) near=.true.
      end do
      if (near) then
        n=nn
        call triangle_rule(corners(:,:,p),n,qn,wn,xp(:,1:n*n),wx(1:n*n))
        call triangle_rule(corners(:,:,s),n,qn,wn,yp(:,1:n*n),wy(1:n*n))
      else if (norm2(centre(:,p)-centre(:,s)) > far_ratio*(size(p)+size(s))) then
        n=nf
        call triangle_rule(corners(:,:,p),n,qf,wf,xp(:,1:n*n),wx(1:n*n))
        call triangle_rule(corners(:,:,s),n,qf,wf,yp(:,1:n*n),wy(1:n*n))
      else
        n=nq
        call triangle_rule(corners(:,:,p),n,q,w,xp(:,1:n*n),wx(1:n*n))
        call triangle_rule(corners(:,:,s),n,q,w,yp(:,1:n*n),wy(1:n*n))
      end if
      le=0; ke=0
      do c=1,n*n
        x=xp(:,c)
        if(p==s) then
          do d=1,3
            va=corners(:,d,p)-x
            vb=corners(:,mod(d,3)+1,p)-x
            jac=norm2(cross(va,vb))
            do u=1,nn
              do v=1,nn
                y=x+qn(u)*((1-qn(v))*va+qn(v)*vb)
                weight=wx(c)*jac*qn(u)*wn(u)*wn(v)
                call interaction(x,y,weight,corners(:,:,p),corners(:,:,s), &
                  factors(:,p),factors(:,s),[n0,n1],.true.,le,ke)
              end do
            end do
          end do
        else
          do d=1,n*n
            call interaction(x,yp(:,d),wx(c)*wy(d),corners(:,:,p),corners(:,:,s), &
              factors(:,p),factors(:,s),[n0,n1],.false.,le,ke)
          end do
        end if
      end do
      if(p==s) then
        do u=1,2
          le(:,:,u)=(le(:,:,u)+transpose(le(:,:,u)))/2
        end do
      end if
      do a=1,3
        i=edge_ids(a,p)
        do b=1,3
          j=edge_ids(b,s)
          values=[le(a,b,1)+le(a,b,2), -(ke(a,b,1)+ke(a,b,2)), ke(a,b,1)+ke(a,b,2), &
                  n0**2*le(a,b,1)+n1**2*le(a,b,2)]
          rows=[i,i,ne+i,ne+i]; cols=[j,ne+j,j,ne+j]
          do k=1,4
            !$omp atomic
            parts(1,rows(k),cols(k))=parts(1,rows(k),cols(k))+real(values(k),dp)
            !$omp atomic
            parts(2,rows(k),cols(k))=parts(2,rows(k),cols(k))+aimag(values(k))
          end do
          if(p/=s) then
            ! Transposed block: rows and columns swap, the same four values.
            rows=[j,j,ne+j,ne+j]; cols=[i,ne+i,i,ne+i]
            do k=1,4
              !$omp atomic
              parts(1,rows(k),cols(k))=parts(1,rows(k),cols(k))+real(values(k),dp)
              !$omp atomic
              parts(2,rows(k),cols(k))=parts(2,rows(k),cols(k))+aimag(values(k))
            end do
          end if
        end do
      end do
    end do
  end do
  !$omp end parallel do
end subroutine

subroutine bem_fields(nt,ne,np,nq,nf,corners,edge_ids,factors,coefficients,points,faces, &
                      index,side,q,w,qf,wf,far_ratio,electric,magnetic)
  use reference_math
  implicit none
  integer, intent(in) :: nt,ne,np,nq,nf,edge_ids(3,nt),faces(np),side
  real(dp), intent(in) :: corners(3,3,nt),factors(3,nt),points(3,np),index,q(nq),w(nq)
  real(dp), intent(in) :: qf(nf),wf(nf),far_ratio
  complex(dp), intent(in) :: coefficients(2*ne)
  complex(dp), intent(out) :: electric(3,np),magnetic(3,np)
  real(dp) :: qp(3,max(nq,nf)**2),qw(max(nq,nf)**2),x(3),y(3),d(3),r,norm(3),va(3),vb(3)
  real(dp) :: jac,weight,edge(3),conormal(3),wave,centre(3,nt),size(nt)
  complex(dp) :: scalar,gradient(3),moment(3),green,radial,jx(3),mx(3),js,ms
  complex(dp) :: aj(3),am(3),gj(3),gm(3),cj(3),cm(3),line_integral
  integer :: p,s,a,b,c,ee,m
  electric=0; magnetic=0; wave=k0*index
  do s=1,nt
    centre(:,s)=sum(corners(:,:,s),dim=2)/3
    size(s)=0
    do a=1,3
      size(s)=max(size(s),norm2(corners(:,a,s)-centre(:,s)))
    end do
  end do
  !$omp parallel do schedule(dynamic,16) default(none) &
  !$omp shared(nt,ne,np,nq,nf,corners,edge_ids,factors,coefficients,points,faces, &
  !$omp        side,q,w,qf,wf,far_ratio,electric,magnetic,wave,centre,size,index) &
  !$omp private(p,s,a,b,c,ee,m,qp,qw,x,y,d,r,norm,va,vb,jac,weight,edge,conormal, &
  !$omp         scalar,gradient,moment,green,radial,jx,mx,js,ms,aj,am,gj,gm,cj,cm, &
  !$omp         line_integral)
  do p=1,np
    x=points(:,p)
    do s=1,nt
      scalar=0; gradient=0; moment=0
      if(faces(p)==s) then
        norm=cross(corners(:,2,s)-corners(:,1,s),corners(:,3,s)-corners(:,1,s))
        norm=norm/norm2(norm)
        do a=1,3
          va=corners(:,a,s)-x; vb=corners(:,mod(a,3)+1,s)-x
          jac=norm2(cross(va,vb))
          do b=1,nq
            do c=1,nq
              y=x+q(b)*((1-q(c))*va+q(c)*vb)
              d=x-y; r=norm2(d)
              weight=jac*q(b)*w(b)*w(c)
              green=weight*exp(ii*wave*r)/(4*pi*r)
              scalar=scalar+green; moment=moment-d*green
            end do
          end do
          edge=corners(:,mod(a,3)+1,s)-corners(:,a,s)
          conormal=cross(edge,norm)/norm2(edge)
          line_integral=0
          do b=1,nq
            y=corners(:,a,s)+q(b)*edge
            r=norm2(x-y)
            line_integral=line_integral+w(b)*norm2(edge)*exp(ii*wave*r)/(4*pi*r)
          end do
          gradient=gradient-conormal*line_integral
        end do
      else
        if (norm2(x-centre(:,s)) > far_ratio*size(s)) then
          m=nf
          call triangle_rule(corners(:,:,s),m,qf,wf,qp(:,1:m*m),qw(1:m*m))
        else
          m=nq
          call triangle_rule(corners(:,:,s),m,q,w,qp(:,1:m*m),qw(1:m*m))
        end if
        do a=1,m*m
          d=x-qp(:,a); r=norm2(d)
          green=qw(a)*exp(ii*wave*r)/(4*pi*r)
          radial=green*(ii*wave-1/r)/r
          scalar=scalar+green; gradient=gradient+radial*d; moment=moment-d*green
        end do
      end if
      jx=0; mx=0; js=0; ms=0
      do a=1,3
        ee=edge_ids(a,s)
        jx=jx+coefficients(ee)*factors(a,s)*(x-corners(:,a,s))
        mx=mx+coefficients(ne+ee)*factors(a,s)*(x-corners(:,a,s))
        js=js+coefficients(ee)*factors(a,s)
        ms=ms+coefficients(ne+ee)*factors(a,s)
      end do
      aj=jx*scalar+js*moment; am=mx*scalar+ms*moment
      gj=2*js*gradient; gm=2*ms*gradient
      cj=[gradient(2)*jx(3)-gradient(3)*jx(2),gradient(3)*jx(1)-gradient(1)*jx(3), &
          gradient(1)*jx(2)-gradient(2)*jx(1)]
      cm=[gradient(2)*mx(3)-gradient(3)*mx(2),gradient(3)*mx(1)-gradient(1)*mx(3), &
          gradient(1)*mx(2)-gradient(2)*mx(1)]
      if(faces(p)==s) then
        gj=gj-side*norm*js; gm=gm-side*norm*ms
        cj=cj-side*0.5_dp*[norm(2)*jx(3)-norm(3)*jx(2),norm(3)*jx(1)-norm(1)*jx(3), &
                          norm(1)*jx(2)-norm(2)*jx(1)]
        cm=cm-side*0.5_dp*[norm(2)*mx(3)-norm(3)*mx(2),norm(3)*mx(1)-norm(1)*mx(3), &
                          norm(1)*mx(2)-norm(2)*mx(1)]
      end if
      electric(:,p)=electric(:,p)+side*(ii*k0*(aj+gj/wave**2)-cm)
      magnetic(:,p)=magnetic(:,p)+side*(cj+ii*k0*index**2*(am+gm/wave**2))
    end do
  end do
  !$omp end parallel do
end subroutine
