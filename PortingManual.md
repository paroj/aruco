# API changes compared to ArUco 1.3

1. `Marker` uses stack allocated `cv::Vec3f` instead of `cv::Mat` for `Rvec, Tvec`
2. `Board` uses stack allocated `cv::Vec3f` instead of `cv::Mat` for `Rvec, Tvec`
3. `FiducidalMarkers::createBoardImage*`: do not internally call `srand` any more
4. `FiducidalMarkers::createBoardImage*`: default parameter `excludedIds` is now an empty vector instead of a `NULL` pointer
5. removed Deprecated Functions
  - `MarkerDetector::enableErosion`: was noop
  - `MarkerDetector::glGetProjectionMatrix`: use CameraParameters instead
  - `MarkerDetector::pyrDown`: was noop
  - `BoardDetector::setYPerperdicular`: use `setYPerpendicular`
6. removed `using namespace std;` from headers
7. made `HighlyReliableMarkers::BalancedBinaryTree` private