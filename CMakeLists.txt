cmake_minimum_required(VERSION 3.5)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message("Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

project(vulkan_triangle)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(vulkan_triangle
  ${PROJECT_SOURCE_DIR}/main.cpp
)

set_property(TARGET vulkan_triangle PROPERTY CXX_STANDARD 20)

target_include_directories(vulkan_triangle
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vulkan_triangle
  glfw
  Vulkan::Vulkan
)
