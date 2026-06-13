set(PPP_MAIN_SRC
  main.c
  )

# Add user code source files here
set(PPP_CODE_SRC
  single.c
  parallel.c
  )

add_library(nbody_parts
  ${PPP_CODE_SRC}
  )

add_executable(nbody ${PPP_MAIN_SRC})
target_link_libraries(nbody
  nbody_parts
  ppp_pnm
  ${MPI_C_LIBRARIES}
  m
  )
