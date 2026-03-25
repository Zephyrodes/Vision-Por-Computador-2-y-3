cmake_minimum_required(VERSION 3.10)

project(TallerC2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Ejecutable
add_executable(Taller main.cpp)

# Incluir headers
target_include_directories(Taller PRIVATE ${OpenCV_INCLUDE_DIRS})

# Linkear librerías
target_link_libraries(Taller ${OpenCV_LIBS})
