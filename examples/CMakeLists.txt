# Each source file is a standalone example with its own main().
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cc" "*.cpp" "*.cxx")
foreach(source IN LISTS EXAMPLE_SOURCES)
    get_filename_component(name "${source}" NAME_WE)
    add_executable(example_${name} "${source}")
    target_link_libraries(example_${name} PRIVATE transformer::transformer)
endforeach()
