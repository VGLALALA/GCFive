void BallImageProc::Unproject3dBallTo2dImage(const cv::Mat& src3D, cv::Mat& destination_image_gray, const GolfBall& ball) {

        // TBD - We already essentially have a 2D Mat.  So why spend all this time copying?
        // Can we just go on to use the 3D Mat?
        // Currently, this function is only used when we need to display one of the 3D projections.
        for (int x = 0; x < destination_image_gray.cols; x++) {
            for (int y = 0; y < destination_image_gray.rows; y++) {
                int position[]{ x, y };
                // There is only one Z-plane in the reduced image - at z = 0
                // The reduced image is a set of uints, so we seem to need to normalize to 0-255 - TBD - why??
                int maxValueZ = src3D.at<cv::Vec2i>(x, y)[0];
                int pixelValue = src3D.at<cv::Vec2i>(x, y)[1];

                int original_pixel_value = (int)destination_image_gray.at<uchar>(x, y);
                /* ONLY FOR DEBUG - TBD
                if (pixelValue != original_pixel_value) {
                    GS_LOG_TRACE_MSG(trace, "Unproject3dBallTo2dImage found different pixel value of " + std::to_string(pixelValue) +
                        " (was " + std::to_string(original_pixel_value) + ") at( " + std::to_string(x) + ", " + std::to_string(y) + ").");
                }
                // std::cout << "pixel from 3D image: " << (int)pixelValue << std::endl;
                */
                destination_image_gray.at<uchar>(x, y) = pixelValue;  // was uchar

                // FOR DEBUG ONLY
                /* ONLY FOR DEBUG - TBD
                if (ball.PointIsInsideBall(x, y) && pixelValue == kPixelIgnoreValue) {
                    GS_LOG_TRACE_MSG(trace, "Unproject3dBallTo2dImage found ignore pixel within ball at (" + std::to_string(x) + ", " + std::to_string(y) + ").");
                }
                */
            }
        }

        // LoggingTools::DebugShowImage("destination_image_gray", destination_image_gray);
        // We're trying to fill in holes here, but this may be fuzzing up the picture too much
        // See if there is a better morphology or interpolation or something
        // TBD- BAD???cv::morphologyEx(destination_image_gray, destination_image_gray, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        
        /**** All of the following attempts for hole-filling have failed:
        LoggingTools::DebugShowImage("(open) destination_image_gray", destination_image_gray);

        cv:: Mat kernel = (cv::Mat_<char>(3, 3) << -1, -1, -1, 
                                             -1,  1, -1, 
                                             -1, -1, -1);

        cv::Mat single_pixels;
        cv::morphologyEx(destination_image_gray, single_pixels, cv::MORPH_HITMISS, kernel);
        LoggingTools::DebugShowImage("single_pixels", single_pixels);
        cv::Mat single_pixels_inv;
        cv::bitwise_not(single_pixels, single_pixels_inv);
        LoggingTools::DebugShowImage("single_pixels_inv", single_pixels_inv);
        cv::bitwise_and(destination_image_gray, destination_image_gray, destination_image_gray, single_pixels_inv);
        LoggingTools::DebugShowImage("(closed) destination_image_gray", destination_image_gray);
        

        OR-----------------

        cv::Mat destination_image_grayComplement;
        cv::bitwise_not(destination_image_gray, destination_image_grayComplement);
        LoggingTools::DebugShowImage("destination_image_grayComplement", destination_image_grayComplement);

        int kernel1Data[9] = { 0, 0, 0,
                               0, 1, 0,
                               0, 0, 0 };
        cv::Mat kernel1 = cv::Mat(3, 3, CV_8U, kernel1Data);

        int kernel2Data[9] = { 1, 1, 1,
                               1, 0, 1,
                               1, 1, 1 };
        cv::Mat kernel2 = cv::Mat(3, 3, CV_8U, kernel2Data);

        cv::Mat hitOrMiss1;
        cv::morphologyEx(destination_image_gray, hitOrMiss1, cv::MORPH_HITMISS, kernel2);
        destination_image_gray = hitOrMiss1;
        /*
        cv::morphologyEx(destination_image_gray, hitOrMiss1, cv::MORPH_ERODE, kernel1);
        LoggingTools::DebugShowImage("hitOrMiss1", hitOrMiss1);
        cv::Mat hitOrMiss2;
        cv::morphologyEx(destination_image_grayComplement, hitOrMiss2, cv::MORPH_ERODE, kernel2);
        LoggingTools::DebugShowImage("hitOrMiss2", hitOrMiss2);
        cv::bitwise_and(hitOrMiss1, hitOrMiss2, destination_image_gray);
        */

        // LoggingTools::DebugShowImage("(closed) destination_image_gray", destination_image_gray);
    }