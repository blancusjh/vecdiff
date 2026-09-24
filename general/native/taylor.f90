! Normalized, total-degree Taylor convolution. Python supplies the index map.
subroutine taylor_product(a, b, left, right, starts, result, np, nc, nt)
  implicit none
  integer, intent(in) :: np, nc, nt
  integer, intent(in) :: left(nt), right(nt), starts(nc+1)
  complex(kind=8), intent(in) :: a(np,nc), b(np,nc)
  complex(kind=8), intent(out) :: result(np,nc)
  integer :: i, j, p, l, r
  result = (0.0d0,0.0d0)
  do i=1,nc
    do j=starts(i),starts(i+1)-1
      l=left(j)
      r=right(j)
      do p=1,np
        result(p,i)=result(p,i)+a(p,l)*b(p,r)
      end do
    end do
  end do
end subroutine taylor_product
