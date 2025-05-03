cv::Mat BallImageProc::ReduceReflections(const cv::Mat& img, const cv::Mat& mask) {

        int hh = img.rows;
        int ww = img.cols;

        LoggingTools::DebugShowImage("ReduceReflections - input img = ", img);
        LoggingTools::DebugShowImage("ReduceReflections - mask = ", mask);

        // threshold

        GsColorTriplet lower{ kReflectionMinimumRGBValue,kReflectionMinimumRGBValue,kReflectionMinimumRGBValue };
        GsColorTriplet upper{ 255,255,255 };

        cv::Mat thresh(img.rows, img.cols, img.type(), cv::Scalar(0));
        cv::inRange(img, lower, upper, thresh);

        LoggingTools::DebugShowImage("ReduceReflections - thresholded image = ", thresh);

        // apply morphology close and open to make mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::Mat morph;
        cv::morphologyEx(thresh, morph, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), /*iterations = */ 1);

        kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));   // originally 25,25
        cv::morphologyEx(morph, morph, cv::MORPH_DILATE, kernel, cv::Point(-1, -1),  /*iterations = */ 1);

        // Now re-apply the appropriate mask outside the circle to ensure that those pixels are not considered, given
        // that some of the regions may have been broadened outside the ball area
        cv::bitwise_and(morph, mask, morph);

        LoggingTools::DebugShowImage("ReduceReflections - morphology = ", morph);

        // use mask with input to do inpainting of the bright bits
        // TBD - What radius to use?  Currently 101 was just a guess?
        cv::Mat result1;
        int inPaintRadius = (int)(std::min(ww, hh) / 30);
        cv::inpaint(img, morph, result1, inPaintRadius, cv::INPAINT_TELEA);
        LoggingTools::DebugShowImage("ReduceReflections - result1 (INPAINT_TELEA) (radius=" + std::to_string(inPaintRadius) + ") = ", result1);

        return result1;
    }