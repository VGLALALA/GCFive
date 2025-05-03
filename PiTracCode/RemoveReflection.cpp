const int kReflectionMinimumRGBValue = 245;  // Nominal is 235.  TBD - Not used - remove?

    void BallImageProc::RemoveReflections(const cv::Mat& original_image, cv::Mat& filtered_image, const cv::Mat& mask) {

        int hh = original_image.rows;
        int ww = original_image.cols;

        static int imgNumber = 1;
        // LoggingTools::DebugShowImage("RemoveReflections - input img# " + std::to_string(imgNumber) + " = ", original_image);
        // LoggingTools::DebugShowImage("filtered_image - input img# " + std::to_string(imgNumber) + " = ", filtered_image);
        imgNumber++;

        // LoggingTools::DebugShowImage("RemoveReflections - mask = ", mask);

        // Define the idea of a "bright" relfection dynamically.  The reflection brightness will be in the
        // xx% percentile (e.g., above 98%)
        // Dynamically determine the reflection minimum based on the other values on the
        // golf ball.  Basically figure out "bright" based on being on the high side of the histogram
        const int brightness_percentage = 99;
        int brightness_cutoff;
        int lowestBrightess;
        int highest_brightness;
        GetImageCharacteristics(original_image, brightness_percentage, brightness_cutoff, lowestBrightess, highest_brightness);

        GS_LOG_TRACE_MSG(trace, "Lower cutoff for brightness is " + std::to_string(brightness_percentage) + "%, grayscale value = " + std::to_string(brightness_cutoff));

        brightness_cutoff--;  // Make sure we don't filter out EVERYTHING
        // GsColorTriplet lower = ((uchar)brightness_cutoff, (uchar)brightness_cutoff, (uchar)brightness_cutoff);
        GsColorTriplet lower = ((uchar)kReflectionMinimumRGBValue, (uchar)kReflectionMinimumRGBValue, (uchar)kReflectionMinimumRGBValue);
        GsColorTriplet upper{ 255,255,255 };

        cv::Mat thresh(original_image.rows, original_image.cols, original_image.type(), cv::Scalar(0));
        cv::inRange(original_image, lower, upper, thresh);

        // LoggingTools::DebugShowImage("RemoveReflections - Initial thresholded image = ", thresh);

        // Expand the bright reflection areas, because they are likely to be areas where
        // the Gabor filters will show a lot of edges that will otherwise pollute the statistics

        static const int kReflectionKernelDilationSize = 5; // Nominal was 25?

        const int kCloseKernelSize = 3;  // 7

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kCloseKernelSize, kCloseKernelSize));
        // Morph is a binary (0 or 255) mask
        cv::Mat morph;
        cv::morphologyEx(thresh, morph, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), /*iterations = */ 1);

        kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kReflectionKernelDilationSize, kReflectionKernelDilationSize));   // originally 25,25
        cv::morphologyEx(morph, morph, cv::MORPH_DILATE, kernel, cv::Point(-1, -1),  /*iterations = */ 1);

        // LoggingTools::DebugShowImage("RemoveReflections - Expanded thresholded image = ", morph);

        // Iterate through the morphed, expanded mask image and set the corresponding pixels to "ignore" in the filtered_image
        for (int x = 0; x < original_image.cols; x++) {
            for (int y = 0; y < original_image.rows; y++) {
                uchar p1 = morph.at<uchar>(x, y);

                if (p1 == 255) {
                    filtered_image.at<uchar>(x, y) = kPixelIgnoreValue;
                }
             }
        }

        LoggingTools::DebugShowImage("RemoveReflections - final filtered image = ", filtered_image);
    }