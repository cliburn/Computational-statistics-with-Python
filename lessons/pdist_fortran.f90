
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

subroutine pdist_fortran(n, p, xs, D)
integer, intent(in) :: n, p
real(8), dimension(n, p), intent(in) :: xs
real(8), dimension(n, n), intent(inout) :: D
integer :: i, j, k
real(8) :: s, tmp
do j = 1, n
    do i = 1, n
        s = 0.0
        do k = 1, p
            tmp = xs(i, k) - xs(j, k)
            s = s + tmp*tmp
        end do
        D(i, j) = sqrt(s)
    end do
end do
end subroutine