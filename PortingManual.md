# API changes compared to ArUco 1.3

1. `Marker` and `Board` uses `double` for `Rvec, Tvec`
3. `FiducidalMarkers::createBoardImage*`: do not internally call `srand` any more
4. `FiducidalMarkers::createBoardImage*`: default parameter `excludedIds` is now an empty vector instead of a `NULL` pointer
5. removed Deprecated Functions
  - `MarkerDetector::enableErosion`: was noop
  - `MarkerDetector::glGetProjectionMatrix`: use CameraParameters instead
  - `MarkerDetector::pyrDown`: was noop
  - `BoardDetector::setYPerperdicular`: use `setYPerpendicular`
6. removed `using namespace std;` from headers
7. made `HighlyReliableMarkers::BalancedBinaryTree` private
8. `MarkerCode::set` now takes full 2d bitcode arrays instead of individual bits
9. `MarkerInfo` class is gone
10. `BoardConfiguration` is no longer inherits of vector
   - use either `BoardConfiguration::objPoints`
   - or `BoardConfiguration::ids` (also replaces `BoardConfiguration::getIdList`)
11. `BoardConfiguration::getIndexOfMarkerId` is gone. use `std::find` and `BoardConfiguration::ids`