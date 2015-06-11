FILE(GLOB hdrs "*.h*")
FILE(GLOB srcs "*.c*")


SET(hdrs ${hdrs})
SET(srcs ${srcs})


ADD_LIBRARY(aruco ${srcs} ${hdrs})
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

INSTALL(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
	LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory

INSTALL(FILES ${hdrs}
	DESTINATION include/${PROJECT_NAME}
	COMPONENT main)

