
subroutine f_fort(s, N)
    integer*8, intent(in) :: N
    integer*8, intent(out) :: s
    integer*8 i
    s = 0
    do i = 1, N
        s = s + i*i
    end do
end 