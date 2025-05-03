 cv::Mat BallImageProc::ApplyGaborFilterToBall(const cv::Mat& image_gray, const GolfBall& ball, float & calibrated_binary_threshold, float prior_binary_threshold) {
        // TBD - Not sure we will ever need the ball information?
        CV_Assert( (image_gray.type() == CV_8UC1) );

        cv::Mat img_f32;
        image_gray.convertTo(img_f32, CV_32F, 1.0 / 255, 0);


        // This two-step calculation of the kernel parameters allows us to use the first set in a 
        // testing/playground environment with easier-to-control parameters and then convert as necessary to
        // the final kernal call.  So, DON'T REFACTOR

        // TBD - For equalized images, these numbers are causing too much noise.
        // For the GS camera, am considering  lambda=14, threshold = 4.
#ifdef GS_USING_IMAGE_EQ
        const int kernel_size = 21;
        int pos_sigma = 2;
        int pos_lambda = 6;   // Nominal: 13.  Lambda = 5 and Gamma = 4 or 3 also works well. last was 8
        int pos_gamma = 4;   // Nominal: 4, might try 3
        int pos_th = 60;   // Nominal: 
        int pos_psi = 9;  // Seems to have to be 9 or 27.  Will be multiplied by 3 degrees - CRITICAL - other values do not work at all
        float binary_threshold = 7.;   // *10.  Nominal: 3, might try 4-7
#else
        const int kernel_size = 21; //21;
        int pos_sigma = 2;   // Nominal: 2  (at 30 degree rotation increments)
        int pos_lambda = 6;   // Nominal: 13.  Lambda = 5 and Gamma = 4 or 3 also works well
        int pos_gamma = 4;   // Nominal: 4
        int pos_th = 60;   // Nominal: 
        int pos_psi = 27;  // Will be multiplied by 3 degrees - CRITICAL - other values do not work at all
        float binary_threshold = 8.5;   // *10.  Nominal: 3
#endif
        // Override the starting binary threshold if we have a prior one
        // This prevents the images from looking different simply due to the
        // different thresholds
        if (prior_binary_threshold > 0) {
            binary_threshold = prior_binary_threshold;
        }

        double sig = pos_sigma / 2.0;
        double lm = (double)pos_lambda;
        double th = (double)pos_th * 2;
        double ps = (double)pos_psi * 10.0;
        double gm = (double)pos_gamma / 20.0;   // Nominal:  30

        int white_percent = 0;

        cv::Mat dimpleImg = ApplyTestGaborFilter(img_f32, kernel_size, sig, lm, th, ps, gm, binary_threshold,
            white_percent);

        GS_LOG_TRACE_MSG(trace, "Initial Gabor filter white percent = " + std::to_string(white_percent));

        bool ratheting_threshold_down = (white_percent < kGaborMinWhitePercent);

        // Give it a second go if we're too white or too black and haven't already overridden the binary threshold
        if (prior_binary_threshold < 0 && 
            (white_percent < kGaborMinWhitePercent || white_percent >= kGaborMaxWhitePercent)) {

            // Keep going down or up (depending on the ractchet direction) until we get within a reasonable
            // white-ness range
            while (white_percent < kGaborMinWhitePercent || white_percent >= kGaborMaxWhitePercent) {
                // Try another gabor setting for less/more white

                if (ratheting_threshold_down)
                {
                    if (kGaborMinWhitePercent - white_percent > 5) {
                        binary_threshold = binary_threshold - 1.0F;
                    }
                    else {
                        binary_threshold = binary_threshold - 0.5F;
                    }
                    GS_LOG_TRACE_MSG(trace, "Trying lower gabor binary_threshold setting of " + std::to_string(binary_threshold) + " for better balance.");
                }
                else {
                    if (white_percent - kGaborMaxWhitePercent > 5) {
                        binary_threshold = binary_threshold + 1.0F;
                    }
                    else {
                        binary_threshold = binary_threshold + 0.5F;
                    }
                    GS_LOG_TRACE_MSG(trace, "Trying higher gabor binary_threshold setting of " + std::to_string(binary_threshold) + " for better balance.");
                }

                dimpleImg = ApplyTestGaborFilter(img_f32, kernel_size, sig, lm, th, ps, gm, binary_threshold,
                    white_percent);
                GS_LOG_TRACE_MSG(trace, "Next, refined, Gabor white percent = " + std::to_string(white_percent));

                // If we've gone as far as we can, just return
                if (binary_threshold > 30 || binary_threshold < 2) {
                    GS_LOG_MSG(warning, "Binaary threshold for Gabor filter reached limit of " + std::to_string(binary_threshold));
                    break;
                }

            }

            // Return the final threshold so that the caller can use for subsequent calls
            calibrated_binary_threshold = binary_threshold;

            GS_LOG_TRACE_MSG(trace, "Final Gabor white percent = " + std::to_string(white_percent));
        }

        return dimpleImg;
    }