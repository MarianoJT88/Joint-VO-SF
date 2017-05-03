PROJECT(Joint-VO-SF)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# custom cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#Find dependencies (Eigen is included in MRPT)
FIND_PACKAGE(MRPT REQUIRED base gui opengl obs)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenNI2 REQUIRED)
FIND_PACKAGE(TBB REQUIRED)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${OpenNI2_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})



ADD_LIBRARY(vo_sf_lib
	joint_vo_sf.h
	structs_parallelization.h
	kmeans.cpp
	visualization.cpp
	solver.cpp
	segmentation_background.cpp
	camera.cpp
	camera.h
	datasets.cpp
	datasets.h
	normal_equation.cpp
	opencv_ext.cpp)
	
TARGET_LINK_LIBRARIES(vo_sf_lib
	${MRPT_LIBS}
	${OpenNI2_LIBRARY}
	${OpenCV_LIBS}
    ${TBB_LIBRARIES})
	
		
#To run online with an RGB-D camera	
ADD_EXECUTABLE(VO-SF-Camera 	main_vo_sf_camera.cpp)	
TARGET_LINK_LIBRARIES(VO-SF-Camera 	vo_sf_lib)

		
#To test it with the TUM dataset		
ADD_EXECUTABLE(VO-SF-Datasets 	main_vo_sf_datasets.cpp)
TARGET_LINK_LIBRARIES(VO-SF-Datasets 	vo_sf_lib)
		
		
#To test it with individual Image pairs		
ADD_EXECUTABLE(VO-SF-ImagePair 	main_vo_sf_imagepair.cpp)
TARGET_LINK_LIBRARIES(VO-SF-ImagePair	vo_sf_lib)
		

#To test it with pre-recorded RGB-D sequences		
ADD_EXECUTABLE(VO-SF-ImageSeq 	main_vo_sf_imageseq.cpp)
TARGET_LINK_LIBRARIES(VO-SF-ImageSeq 	vo_sf_lib)
			




# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -mavx")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

