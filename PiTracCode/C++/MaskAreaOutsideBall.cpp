cv::Mat BallImageProc::MaskAreaOutsideBall(cv::Mat& ball_image, const GolfBall& ball, float mask_reduction_factor, const cv::Scalar& maskValue) {

        // LoggingTools::DebugShowImage("MaskAreaOutsideBall - ball_image", ball_image);

        // A white circle on a black background will act as our first mask to preserve the ball portion of the image

        int mask_radius = (int)(ball.measured_radius_pixels_ * mask_reduction_factor);

        cv::Mat maskImage = cv::Mat::zeros(ball_image.rows, ball_image.cols, ball_image.type());
        cv::circle(maskImage, cv::Point(ball.x(), ball.y()), mask_radius, cv::Scalar(255, 255, 255), -1);
        //LoggingTools::DebugShowImage("1st maskImage", maskImage);

        // At this point, maskImage is an image with a white circle and a black outside

        cv::Mat result = ball_image.clone();
        cv::bitwise_and(ball_image, maskImage, result);
        //LoggingTools::DebugShowImage("Intermediate result", result);

        // Now XOR the image-on-black with a on a rectangle of desired color and a black circle in the middle
        cv::Rect r(cv::Point(0, 0), cv::Point(ball_image.cols, ball_image.rows));
        cv::rectangle(maskImage, r, maskValue, cv::FILLED);
        cv::circle(maskImage, cv::Point(ball.x(), ball.y()), mask_radius, cv::Scalar(0, 0, 0), -1);
        //LoggingTools::DebugShowImage("2nd maskImage", maskImage);

        cv::bitwise_xor(result, maskImage, result);

        // LoggingTools::DebugShowImage("MaskAreaOutsideBall: result", result);

        return result;
    }