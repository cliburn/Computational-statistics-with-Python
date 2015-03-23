
subroutine pdist_fortran (n, p, A, D)

    integer, intent(in) :: n
    integer, intent(in) :: p
    real(8), intent(in), dimension(n,p) :: A
    real(8), intent(inout), dimension(n,n) :: D
            
    integer :: i, j, k
    real(8) :: s, tmp
    ! note order of indices is different from C
    do j = 1, n
        do i = 1, n
            s = 0.0
            do k = 1, p
                tmp = A(i, k) - A(j, k)
                s = s + tmp*tmp
            end do
            D(i, j) = sqrt(s)
        end do
    end do
end subroutine