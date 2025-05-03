void BallImageProc::GetRotatedImage(const cv::Mat& gray_2D_input_image, const GolfBall& ball, const cv::Vec3i rotation, cv::Mat& outputGrayImg) {
       BOOST_LOG_FUNCTION();                    
       
       // Project the ball out onto a 3D hemisphere at the current x, y, and z-axis rotation
       // and then unproject back to 2D matrix (image)
       cv::Mat ball3DImage = Project2dImageTo3dBall(gray_2D_input_image, ball, rotation);

       // TBD - FOR DEBUG
       // outputGrayImg = gray_2D_input_image.clone();

       outputGrayImg = cv::Mat::zeros(gray_2D_input_image.rows, gray_2D_input_image.cols, gray_2D_input_image.type());
       Unproject3dBallTo2dImage(ball3DImage, outputGrayImg, ball);
   }