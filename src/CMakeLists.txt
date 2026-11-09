file(GLOB_RECURSE TRANSFORMER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx")

add_library(transformer STATIC ${TRANSFORMER_SOURCES})
add_library(transformer::transformer ALIAS transformer)
target_include_directories(transformer PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(transformer PUBLIC Eigen3::Eigen)
