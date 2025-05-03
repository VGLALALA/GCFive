cv::Mat BallImageProc::IsolateBall(const cv::Mat& img, GolfBall& ball) {

        // We will grab a rectangle a little larger than the actual ball size
        const float ballSurroundMult = 1.05f;

        int r1 = (int)std::round(ball.measured_radius_pixels_ * ballSurroundMult);
        int rInc = (long)(r1 - ball.measured_radius_pixels_);
        // Don't assume the ball is well within the larger picture

        int x1 = ball.x() - r1;
        int y1 = ball.y() - r1;
        int x_width = 2 * r1;
        int y_height = 2 * r1;

        // Ensure the isolated image is entirely in the larger image
        x1 = max(0, x1);
        y1 = max(0, y1);

        if (x1 + x_width >= img.cols) {
            x1 = img.cols - x_width - 1;
        }
        if (y1 + y_height >= img.rows) {
            y1 = img.rows - y_height - 1;
        }

        cv::Rect ballRect{ x1, y1, x_width, y_height };

        // Re-center the ball's x and y position in the new, smaller picture
        // This will change the ball that was sent in
        ball.set_x( (float)std::round(rInc + ball.measured_radius_pixels_));
        ball.set_y( (float)std::round(rInc + ball.measured_radius_pixels_));

        cv::Point offset_sub_to_full;
        cv::Point offset_full_to_sub;
        cv::Mat ball_image = CvUtils::GetSubImage(img, ballRect, offset_sub_to_full, offset_full_to_sub);

        // Draw the mask circle slightly smaller than the ball to prevent any bright prenumbra around the isolated ball
        const float referenceBallMaskReductionFactor = 0.995f;

        // Do equalized images help?
#ifdef GS_USING_IMAGE_EQ
        cv::equalizeHist(ball_image, ball_image);
#endif

        cv::Mat finalResult = MaskAreaOutsideBall(ball_image, ball, referenceBallMaskReductionFactor, cv::Scalar(0, 0, 0));

        // LoggingTools::DebugShowImage("finalResult", finalResult);

        return finalResult;
    }