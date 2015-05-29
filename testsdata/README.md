This directory contains data for testing the aruco library.

# board

 - video.avi: video that shows a board
 - intrinsics.yml: intrinsic information of the camera in the OpenCv format
 - board_pix.yml: configuration of the board shown in the video. It contains the id of the markers in the boards as well as its arrangement in pixels
 - board_meters.yml:same as before, but in meters
 - board.png: printable image of the board
 - image-test.png: a sample image

In order to test the library with these files you can use :

- aruco_simple_board: the simplest example.
    Run it as:

    ` aruco_simple_board image-test.png board_pix.yml intrinsics.yml 0.039`

	or

    ` aruco_simple_board image-test.png board_meters.yml intrinsics.yml`

- aruco_test_board: a more sohpisticated example
    Run it as:

    ` aruco_test_board    video.avi    board_pix.yml  intrinsics.yml 0.039`

    or

    ` aruco_test_board    video.avi    board_meters.yml  intrinsics.yml`
- aruco_test_board_gl: example showing how to use OpenGL
    Run it as:

    ` aruco_test_board_gl   video.avi   board_pix.yml   intrinsics.yml  0.039`

    or

    ` aruco_test_board_gl   video.avi   board_meters.yml   intrinsics.yml`

# chessboard

This video shows the use of chessboards. You can use it with

`aruco_test_board chessboard.avi chessboardinfo_pix.yml intrinsics.yml 0.034`

or

`aruco_test_board chessboard.avi chessboardinfo_meters.yml intrinsics.yml`


Note: (0.034 is the size of the board when was printed. You can print your own chessboard using chessboard.pdf)

# single
The directory contains:
  - video.avi: video showing a piece of paper with six diferent markers
  - intrinsics.yml: file with the intrisic parameters, in YAML format as provided by the calibration.cpp application of OpenCv2.2


In order to test the library with these files you can use :

- aruco_simple: the simplest example.
     Run it as:  ` aruco_simple image-test.png intrinsics.yml 0.05`
- aruco_test: a more sohpisticated example
     Run it as: ` aruco_test  video.avi  intrinsics.yml  0.05`
- aruco_test_gl: example showing how to use OpenGL
     Run it as:  ` aruco_test_gl   video.avi   intrinsics.yml  0.05`


NOTE: The option   0.05 indicates that the real size of the marker's sides in meters (5 cm).
