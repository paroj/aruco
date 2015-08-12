FILE(GLOB hdrs "*.h*")
FILE(GLOB srcs "*.c*")

INCLUDE(GenerateExportHeader)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

ADD_LIBRARY(aruco ${srcs} ${hdrs})
GENERATE_EXPORT_HEADER(aruco EXPORT_MACRO_NAME ARUCO_EXPORTS)
ADD_COMPILER_EXPORT_FLAGS()
target_include_directories(aruco PUBLIC ${PROJECT_SOURCE_DIR}/src/ ${OpenCV_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/src/)
TARGET_LINK_LIBRARIES(aruco PUBLIC ${REQUIRED_LIBRARIES})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR=${OpenCV_LIB_DIR}")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES 				# create *nix style library versions + symbolic links
	DEFINE_SYMBOL DSO_EXPORTS
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_SOVERSION}
	CLEAN_DIRECT_OUTPUT 1 					# allow creating static and shared libs without conflicts
	OUTPUT_NAME "aruco${PROJECT_DLLVERSION}" 	# avoid conflicts between library and binary target names
)

INSTALL(TARGETS aruco
	RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
	LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory

INSTALL(FILES ${hdrs} ${PROJECT_BINARY_DIR}/src/aruco_export.h
	DESTINATION include/aruco
	COMPONENT main)

