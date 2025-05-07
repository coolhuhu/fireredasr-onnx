function(download_gflags)
  include(FetchContent)
  message("Downloading and configuring gflags...")
  
  set(URL "https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz")
  set(URL_HASH "SHA256=34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf")

  set(possible_file_location "${CMAKE_SOURCE_DIR}/third-part/gflags-2.2.2.tar.gz")

  if (EXISTS ${possible_file_location})
    set(URL "${possible_file_location}")
    file(TO_CMAKE_PATH "${URL}" URL)
    message(STATUS "Found local gflags archive: ${URL}")
  endif()

  set(BUILD_SHARED_LIBS ON)
  set(BUILD_STATIC_LIBS OFF)
  set(BUILD_gflags_LIBS ON)
  set(INSTALL_SHARED_LIBS ON)

  FetchContent_Declare(
    gflags
    URL ${URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(gflags)

  message(STATUS "gflags build dir is ${gflags_BINARY_DIR}")

endfunction()

download_gflags()
