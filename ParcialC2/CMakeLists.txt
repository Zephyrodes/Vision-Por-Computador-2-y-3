cmake_minimum_required(VERSION 3.10)

# Nombre del proyecto
project(ParcialC2)

# Estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Incluir directorios
include_directories(${OpenCV_INCLUDE_DIRS})

# Ejecutable
add_executable(ParcialC2 main.cpp)

# Linkear OpenCV
target_link_libraries(ParcialC2 ${OpenCV_LIBS})
