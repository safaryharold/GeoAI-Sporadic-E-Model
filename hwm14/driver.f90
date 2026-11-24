program driver
  ! reads lines: iyd sec alt glat glon stl f107a f107 ap ; writes meridional zonal (m/s)
  implicit none
  integer(4) :: iyd, ios
  real(4) :: sec, alt, glat, glon, stl, f107a, f107, ap(2), w(2)
  ap(1) = -1.0
  do
    read(*,*,iostat=ios) iyd, sec, alt, glat, glon, stl, f107a, f107, ap(2)
    if (ios /= 0) exit
    call hwm14(iyd, sec, alt, glat, glon, stl, f107a, f107, ap, w)
    write(*,'(2F10.3)') w(1), w(2)
  end do
end program driver
