

#include <ranges>
#include <algorithm>
#include <vector>
#include "gs_format_lib.h"

#include <boost/timer/timer.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/circular_buffer.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cvdef.h>

#include "ball_image_proc.h"
#include "logging_tools.h"
#include "cv_utils.h"
#include "gs_config.h"
#include "gs_options.h"
#include "gs_ui_system.h"
#include "EllipseDetectorCommon.h"
#include "EllipseDetectorYaed.h"

// Edge detection
#include "ED.h"
#include "EDPF.h"
#include "EDColor.h"

namespace golf_sim {

    // Currently, equalizing the brightness of the input images appears to help the results
#define GS_USING_IMAGE_EQ
#define DONT__PERFORM_FINAL_TARGETTED_BALL_ID  // Remove DONT to perform a final, targetted refinement of the ball circle identification
#define DONT__USE_ELLIPSES_FOR_FINAL_ID    

    const int MIN_BALL_CANDIDATE_RADIUS = 10;

    // Balls with an average color that is too far from the searched-for color will not be considered
    // good candidates.The tolerance is based on a Euclidian distance. See differenceRGB in CvUtils module.
    // The tolerance is relative to the closest - in - RGB - value candidate.So if the "best" candidate ball is,
    // for example, 100 away from the expected color, than any balls with a RGB difference of greater than
    // 100 + CANDIDATE_BALL_COLOR_TOLERANCE will be excluded.
    const int CANDIDATE_BALL_COLOR_TOLERANCE = 50;

    const bool PREBLUR_IMAGE = false;
    const bool IS_COLOR_MASKING = false;   // Probably not effective on IR pictures

    // May be necessary in brighter environments - TBD
    const bool FINAL_BLUR = true;

    const int MAX_FINAL_CANDIDATE_BALLS_TO_SHOW = 4;


    // See places of use for explanation of these constants
    static const double kColorMaskWideningAmount = 35;
    static const double kEllipseColorMaskWideningAmount = 35;
    static const bool kSerializeOpsForDebug = false;

    int BallImageProc::kCoarseXRotationDegreesIncrement = 6;
    int BallImageProc::kCoarseXRotationDegreesStart = -42;
    int BallImageProc::kCoarseXRotationDegreesEnd = 42;
    int BallImageProc::kCoarseYRotationDegreesIncrement = 5;
    int BallImageProc::kCoarseYRotationDegreesStart = -30;
    int BallImageProc::kCoarseYRotationDegreesEnd = 30;
    int BallImageProc::kCoarseZRotationDegreesIncrement = 6;
    int BallImageProc::kCoarseZRotationDegreesStart = -50;
    int BallImageProc::kCoarseZRotationDegreesEnd = 60;

    double BallImageProc::kPlacedBallCannyLower;
    double BallImageProc::kPlacedBallCannyUpper;
    double BallImageProc::kPlacedBallStartingParam2 = 40;
    double BallImageProc::kPlacedBallMinParam2 = 30;
    double BallImageProc::kPlacedBallMaxParam2 = 60;
    double BallImageProc::kPlacedBallCurrentParam1 = 120.0;
    double BallImageProc::kPlacedBallParam2Increment = 4;

    int BallImageProc::kPlacedMinHoughReturnCircles = 1;
    int BallImageProc::kPlacedMaxHoughReturnCircles = 4;
    double BallImageProc::kStrobedBallsCannyLower = 50;
    double BallImageProc::kStrobedBallsCannyUpper = 110;


    int BallImageProc::kStrobedBallsMaxHoughReturnCircles = 12;
    int BallImageProc::kStrobedBallsMinHoughReturnCircles = 1;

    int BallImageProc::kStrobedBallsPreCannyBlurSize = 5;
    int BallImageProc::kStrobedBallsPreHoughBlurSize = 13;
    double BallImageProc::kStrobedBallsStartingParam2 = 40;
    double BallImageProc::kStrobedBallsMinParam2 = 30;
    double BallImageProc::kStrobedBallsMaxParam2 = 60;
    double BallImageProc::kStrobedBallsCurrentParam1 = 120.0;
    double BallImageProc::kStrobedBallsHoughDpParam1 = 1.5;
    double BallImageProc::kStrobedBallsParam2Increment = 4;

    bool  BallImageProc::kStrobedBallsUseAltHoughAlgorithm = true;
    double BallImageProc::kStrobedBallsAltCannyLower = 35;
    double BallImageProc::kStrobedBallsAltCannyUpper = 70;
    int BallImageProc::kStrobedBallsAltPreCannyBlurSize = 11;
    int BallImageProc::kStrobedBallsAltPreHoughBlurSize = 16;
    double BallImageProc::kStrobedBallsAltStartingParam2 = 0.95;
    double BallImageProc::kStrobedBallsAltMinParam2 = 0.6;
    double BallImageProc::kStrobedBallsAltMaxParam2 = 1.0;
    double BallImageProc::kStrobedBallsAltCurrentParam1 = 130.0;
    double BallImageProc::kStrobedBallsAltHoughDpParam1 = 1.5;
    double BallImageProc::kStrobedBallsAltParam2Increment = 0.05;

    bool BallImageProc::kUseCLAHEProcessing;
    int BallImageProc::kCLAHEClipLimit;
    int BallImageProc::kCLAHETilesGridSize;

    double BallImageProc::kPuttingBallStartingParam2 = 40;
    double BallImageProc::kPuttingBallMinParam2 = 30;
    double BallImageProc::kPuttingBallMaxParam2 = 60;
    double BallImageProc::kPuttingBallCurrentParam1 = 120.0;
    double BallImageProc::kPuttingBallParam2Increment = 4;
    int BallImageProc::kPuttingMaxHoughReturnCircles = 12;
    int BallImageProc::kPuttingMinHoughReturnCircles = 1;
    double BallImageProc::kPuttingHoughDpParam1 = 1.5;

    double BallImageProc::kExternallyStrobedEnvCannyLower = 35;
    double BallImageProc::kExternallyStrobedEnvCannyUpper = 80;

    double BallImageProc::kExternallyStrobedEnvCurrentParam1 = 130.0;
    double BallImageProc::kExternallyStrobedEnvMinParam2 = 28;
    double BallImageProc::kExternallyStrobedEnvMaxParam2 = 100;
    double BallImageProc::kExternallyStrobedEnvStartingParam2 = 65;
    double BallImageProc::kExternallyStrobedEnvNarrowingParam2 = 0.6;
    double BallImageProc::kExternallyStrobedEnvNarrowingDpParam = 1.1;
    double BallImageProc::kExternallyStrobedEnvParam2Increment = 4;
    int BallImageProc::kExternallyStrobedEnvMinHoughReturnCircles = 3;
    int BallImageProc::kExternallyStrobedEnvMaxHoughReturnCircles = 20;
    int BallImageProc::kExternallyStrobedEnvPreHoughBlurSize = 11;
    int BallImageProc::kExternallyStrobedEnvPreCannyBlurSize = 3;
    double BallImageProc::kExternallyStrobedEnvHoughDpParam1 = 1.0;
    int BallImageProc::kExternallyStrobedEnvNarrowingPreCannyBlurSize = 3;
    int BallImageProc::kExternallyStrobedEnvNarrowingPreHoughBlurSize = 9;
    int BallImageProc::kExternallyStrobedEnvMinimumSearchRadius = 60;
    int BallImageProc::kExternallyStrobedEnvMaximumSearchRadius = 80;

    bool BallImageProc::kUseDynamicRadiiAdjustment = true;
    int BallImageProc::kNumberRadiiToAverageForDynamicAdjustment = 3;
    double BallImageProc::kStrobedNarrowingRadiiMinRatio = 0.8;
    double BallImageProc::kStrobedNarrowingRadiiMaxRatio = 1.2;
    double BallImageProc::kStrobedNarrowingRadiiDpParam = 1.8;
    double BallImageProc::kStrobedNarrowingRadiiParam2 = 100.0;


    double BallImageProc::kPlacedNarrowingRadiiMinRatio = 0.9;
    double BallImageProc::kPlacedNarrowingRadiiMaxRatio = 1.1;
    double BallImageProc::kPlacedNarrowingStartingParam2 = 80.0;
    double BallImageProc::kPlacedNarrowingRadiiDpParam = 2.0;
    double BallImageProc::kPlacedNarrowingParam1 = 130.0;

    int BallImageProc::kPlacedPreCannyBlurSize = 5;
    int BallImageProc::kPlacedPreHoughBlurSize = 11;
    int BallImageProc::kPuttingPreHoughBlurSize = 9;


    bool BallImageProc::kLogIntermediateSpinImagesToFile = false;
    double BallImageProc::kPlacedBallHoughDpParam1 = 1.5;

    bool BallImageProc::kUseBestCircleRefinement = false;
    bool BallImageProc::kUseBestCircleLargestCircle = false;

    double BallImageProc::kBestCircleCannyLower = 55;
    double BallImageProc::kBestCircleCannyUpper = 110;
    int BallImageProc::kBestCirclePreCannyBlurSize = 5;
    int BallImageProc::kBestCirclePreHoughBlurSize = 13;
    double BallImageProc::kBestCircleParam1 = 120.;
    double BallImageProc::kBestCircleParam2 = 35.;
    double BallImageProc::kBestCircleHoughDpParam1 = 1.5;

    double BallImageProc::kExternallyStrobedBestCircleCannyLower = 55;
    double BallImageProc::kExternallyStrobedBestCircleCannyUpper = 110;
    int BallImageProc::kExternallyStrobedBestCirclePreCannyBlurSize = 5;
    int BallImageProc::kExternallyStrobedBestCirclePreHoughBlurSize = 13;
    double BallImageProc::kExternallyStrobedBestCircleParam1 = 120.;
    double BallImageProc::kExternallyStrobedBestCircleParam2 = 35.;
    double BallImageProc::kExternallyStrobedBestCircleHoughDpParam1 = 1.5;

    bool BallImageProc::kExternallyStrobedUseCLAHEProcessing = true;
    int BallImageProc::kExternallyStrobedCLAHEClipLimit = 6;
    int BallImageProc::kExternallyStrobedCLAHETilesGridSize = 6;

    double BallImageProc::kBestCircleIdentificationMinRadiusRatio = 0.85;
    double BallImageProc::kBestCircleIdentificationMaxRadiusRatio = 1.10;

    int BallImageProc::kGaborMaxWhitePercent = 44; // Nominal 46;
    int BallImageProc::kGaborMinWhitePercent = 38; // Nominal 40;



    void BallImageProc::GetImageCharacteristics(const cv::Mat& img,
                                                const int brightness_percentage,
                                                int& brightness_cutoff,
                                                int& lowest_brightness,
                                                int& highest_brightness) {


        /// Establish the number of bins
        const int histSize = 256;

        /// Set the ranges ( for B,G,R) )
        float range[] = { 0, 256 };
        const float* histRange = { range };

        bool uniform = true; bool accumulate = false;

        cv::Mat b_hist;

        /// Compute the histograms:
        calcHist(&img, 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

        // Draw the histograms for B, G and R
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound((double)hist_w / histSize);

        /*
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

        // Normalize the result to [ 0, histImage.rows ]
        cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        */

        long totalPoints = img.rows * img.cols;
        long accum = 0;
        int i = histSize - 1;
        bool foundPercentPoint = false;
        highest_brightness = -1;
        double targetPoints = (double)totalPoints * (100 - brightness_percentage) / 100.0;

        while (i >= 0 && !foundPercentPoint )
        {
            int numPixelsInBin = cvRound(b_hist.at<float>(i));
            accum += numPixelsInBin;
            foundPercentPoint = (accum >= targetPoints) ? true : false;
            if (highest_brightness < 0 && numPixelsInBin > 0) {
                highest_brightness = i;
            }
            i--;  // move to the next bin to the left
        }

        brightness_cutoff = i + 1;
    }



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


    // Returns new coordinates in the passed-in ball, so make a copy of it before
    // calling this if the original information needs to be preserved
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


    cv::Vec3d BallImageProc::GetBallRotation(const cv::Mat& full_gray_image1, 
                                             const GolfBall& ball1, 
                                             const cv::Mat& full_gray_image2, 
                                             const GolfBall& ball2) {
        // NOTE - This function (and downstream functions) assumes that ball1 is the earlier-in-time ball
        // for a right-handed shot.  So, for example, the expected spin will be largely counter-clockwise
        // from ball 1 to ball 2.
        // Make sure that for left-handed shots this is correct - we will assume that for
        // left-handed shots, ball1 is still to the LEFT of ball 2
        
        BOOST_LOG_FUNCTION();

        GS_LOG_TRACE_MSG(trace, "GetBallRotation called with ball1 = " + ball1.Format() + ",\nball2 = " + ball2.Format());
        LoggingTools::DebugShowImage("full_gray_image1", full_gray_image1);
        LoggingTools::DebugShowImage("full_gray_image2", full_gray_image2);

        // First, get a clean picture of each ball with nothing in the background, both sized the exactly same way 
        // Resize the images so that the balls are the same radius. 

        GolfBall local_ball1 = ball1;
        GolfBall local_ball2 = ball2;


        // NOTE - The ball that is passed into the IsolateBall image will be adjusted
        // to have the new x, y, and radius values relative to the smaller, isolated picture
        cv::Mat ball_image1 = IsolateBall(full_gray_image1, local_ball1);
        cv::Mat ball_image2 = IsolateBall(full_gray_image2, local_ball2);

        LoggingTools::DebugShowImage("ISOLATED full_gray_image1", ball_image1);
        LoggingTools::DebugShowImage("ISOLATED full_gray_image2", ball_image2);

        if (GolfSimOptions::GetCommandLineOptions().artifact_save_level_ != ArtifactSaveLevel::kNoArtifacts && kLogIntermediateSpinImagesToFile) {
            LoggingTools::LogImage("", ball_image1, std::vector < cv::Point >{}, true, "log_view_ISOLATED_full_gray_image1.png");
            LoggingTools::LogImage("", ball_image2, std::vector < cv::Point >{}, true, "log_view_ISOLATED_full_gray_image2.png");
        }

        // Just to test.  Ignore the 0 bin
        // CvUtils::DrawGrayImgHistogram(ball_image1, true);


        // We will assume that the images are now square

        double ball1RadiusMultiplier = 1.0;
        double ball2RadiusMultiplier = 1.0;

        if (ball_image1.rows > ball_image2.rows || ball_image1.cols > ball_image2.cols) {
            ball2RadiusMultiplier = (double)ball_image1.rows / (double)ball_image2.rows;
            int upWidth = ball_image1.cols;
            int upHeight = ball_image1.rows;
            cv::resize(ball_image2, ball_image2, cv::Size(upWidth, upHeight), cv::INTER_LINEAR);
        }
        else if (ball_image2.rows > ball_image1.rows || ball_image2.cols > ball_image1.cols) {
            ball1RadiusMultiplier = (double)ball_image2.rows / (double)ball_image1.rows;
            int upWidth = ball_image2.cols;
            int upHeight = ball_image2.rows;
            cv::resize(ball_image1, ball_image1, cv::Size(upWidth, upHeight), cv::INTER_LINEAR);
        }

        // Save the original, non-equalized images for later QA
        cv::Mat originalBallImg1 = ball_image1.clone();
        cv::Mat originalBallImg2 = ball_image2.clone();

        // Adjust relevant ball radius information accordingly
        local_ball1.measured_radius_pixels_ = local_ball1.measured_radius_pixels_ * ball1RadiusMultiplier;
        local_ball1.ball_circle_[2] = local_ball1.ball_circle_[2] * (float)ball1RadiusMultiplier;
        local_ball1.set_x( (float)((double)local_ball1.x() * ball1RadiusMultiplier));
        local_ball1.set_y( (float)((double)local_ball1.y() * ball1RadiusMultiplier));
        local_ball2.measured_radius_pixels_ = local_ball2.measured_radius_pixels_ * ball2RadiusMultiplier;
        local_ball2.ball_circle_[2] = local_ball2.ball_circle_[2] * (float)ball2RadiusMultiplier;
        local_ball2.set_x( (float)((double)local_ball2.x() * ball2RadiusMultiplier));
        local_ball2.set_y( (float)((double)local_ball2.y() * ball2RadiusMultiplier));


        std::vector < cv::Point > center1 = { cv::Point{(int)local_ball1.x(), (int)local_ball1.y()} };
        LoggingTools::DebugShowImage("Ball1 Image", ball_image1, center1);
        GS_LOG_TRACE_MSG(trace, "Updated (local) ball1 data: " + local_ball1.Format());
        std::vector < cv::Point > center2 = { cv::Point{(int)local_ball2.x(), (int)local_ball2.y()} };
        LoggingTools::DebugShowImage("Ball2 Image", ball_image2, center2);
        GS_LOG_TRACE_MSG(trace, "Updated (local) ball2 data: " + local_ball2.Format());

        float calibrated_binary_threshold = 0;
        cv::Mat ball_image1DimpleEdges = ApplyGaborFilterToBall(ball_image1, local_ball1, calibrated_binary_threshold);
        //  Suggest the same binary threshold between the images as a starting point for the second ball - they are probably similar
        cv::Mat ball_image2DimpleEdges = ApplyGaborFilterToBall(ball_image2, local_ball2, calibrated_binary_threshold, calibrated_binary_threshold);
   
        // TBD = Consider inverting the image to focus only on the inner parts of the dimples that will
        // have fewer pixels?
        //cv::bitwise_not(ball_image1, ball_image1);
        //cv::bitwise_not(ball_image2, ball_image2);

        // LoggingTools::DebugShowImage("Ball1 Dimple Image", ball_image1DimpleEdges);
        // LoggingTools::DebugShowImage("Ball2 Dimple Image", ball_image2DimpleEdges);

        cv::Mat area_mask_image_;
        RemoveReflections(ball_image1, ball_image1DimpleEdges, area_mask_image_);
        RemoveReflections(ball_image2, ball_image2DimpleEdges, area_mask_image_);

        // TBD - In addition to removing reflections, we may also want to remove really dark areas which will
        // comprise the registration marks.  That seems counter-intuitive, but those marks sometimes create large
        // "positive" (on) areas in the Gabor filters

        // The outer edge of the ball doesn't provide much information, so ignore it
        const float finalBallMaskReductionFactor = 0.92f;
        cv::Scalar ignoreColor = cv::Scalar(kPixelIgnoreValue, kPixelIgnoreValue, kPixelIgnoreValue);
        ball_image1DimpleEdges = MaskAreaOutsideBall(ball_image1DimpleEdges, local_ball1, finalBallMaskReductionFactor, ignoreColor);
        ball_image2DimpleEdges = MaskAreaOutsideBall(ball_image2DimpleEdges, local_ball2, finalBallMaskReductionFactor, ignoreColor);
        LoggingTools::DebugShowImage("Final ball_image1DimpleEdges after masking outside", ball_image1DimpleEdges);
        LoggingTools::DebugShowImage("Final ball_image2DimpleEdges after masking outside", ball_image2DimpleEdges);

        // Finally, rotate the second ball image to make up for the angle imparted by any offset of the ball from the
        // center of the camera's view.  Just reset the view using the angle offsets from the camera's perspective
        cv::Vec3d ball2Distances;

        // Find the differences between the offset angles, as they may be similar.
        // These will be the angles that the image will have to be rotated in order
        // to make it appear as it would if it were in the center of the image
        cv::Vec3f angleOffset1 = cv::Vec3f((float)ball1.angles_camera_ortho_perspective_[0], (float)ball1.angles_camera_ortho_perspective_[1], 0);
        cv::Vec3f angleOffset2 = cv::Vec3f((float)ball2.angles_camera_ortho_perspective_[0], (float)ball2.angles_camera_ortho_perspective_[1], 0);


        // We will split the difference in the angles so that the amount of de-rotation we need to do is spread evenly
        // across the two images

        // angleOffsetDeltas1 (and the floating-point version) are the angles that ball 1 must be rotated in
        // order to take it halfway to where ball 2 is
        cv::Vec3f angleOffsetDeltas1Float = (angleOffset2 - angleOffset1) / 2.0;
        angleOffsetDeltas1Float[1] *= -1.0;  // Account for how our rotations are signed
        cv::Vec3i angleOffsetDeltas1 = CvUtils::Round(angleOffsetDeltas1Float);


        cv::Mat unrotatedBallImg1DimpleEdges = ball_image1DimpleEdges.clone();
        GetRotatedImage(unrotatedBallImg1DimpleEdges, local_ball1, angleOffsetDeltas1, ball_image1DimpleEdges);

        GS_LOG_TRACE_MSG(trace, "Adjusting rotation for camera view of ball 1 to offset (x,y,z)=" + std::to_string(angleOffsetDeltas1[0]) + "," + std::to_string(angleOffsetDeltas1[1]) + "," + std::to_string(angleOffsetDeltas1[2]));
        LoggingTools::DebugShowImage("Final perspective-de-rotated filtered ball_image1DimpleEdges: ", ball_image1DimpleEdges, center1);
        
        // The second rotation deltas will be the remainder of (approximately) the other half of the necessary degrees to get everything to be the same perspective
        cv::Vec3i angleOffsetDeltas2 = CvUtils::Round(  -(( angleOffset2 - angleOffset1) - angleOffsetDeltas1Float) );
        angleOffsetDeltas2[1] = -angleOffsetDeltas2[1];


        cv::Mat unrotatedBallImg2DimpleEdges = ball_image2DimpleEdges.clone();
        GetRotatedImage(unrotatedBallImg2DimpleEdges, local_ball2, angleOffsetDeltas2, ball_image2DimpleEdges);
        GS_LOG_TRACE_MSG(trace, "Adjusting rotation for camera view of ball 2 to offset (x,y,z)=" + std::to_string(angleOffsetDeltas2[0]) + "," + std::to_string(angleOffsetDeltas2[1]) + "," + std::to_string(angleOffsetDeltas2[2]));
        LoggingTools::DebugShowImage("Final perspective-de-rotated filtered ball_image2DimpleEdges: ", ball_image2DimpleEdges, center1);

        // Although unnecessary for the algorithm, the following DEBUG code shows the original image as it would appear rotated in the same way as the Gabor-filtered balls
        
        cv::Mat normalizedOriginalBallImg1 = originalBallImg1.clone();
        GetRotatedImage(originalBallImg1, local_ball1, angleOffsetDeltas1, normalizedOriginalBallImg1);
        LoggingTools::DebugShowImage("Final rotated originalBall1: ", normalizedOriginalBallImg1, center1);
        cv::Mat normalizedOriginalBallImg2 = originalBallImg2.clone();
        GetRotatedImage(originalBallImg2, local_ball2, angleOffsetDeltas2, normalizedOriginalBallImg2);
        LoggingTools::DebugShowImage("Final rotated originalBall2: ", normalizedOriginalBallImg2, center2);
        
#ifdef __unix__ 
        // Save the normalized ball images to the webserver shared directory so that the user
        // can compare them to the final rotated image.
        GsUISystem::SaveWebserverImage(GsUISystem::kWebServerResultSpinBall1Image, normalizedOriginalBallImg1);
        GsUISystem::SaveWebserverImage(GsUISystem::kWebServerResultSpinBall2Image, normalizedOriginalBallImg2);
#endif



        // Now compute all the possible rotations of the first image so we can figure out which angles make it look like the second ball image
        RotationSearchSpace initialSearchSpace;

        // Initial angle search will be fairly coarse
        initialSearchSpace.anglex_rotation_degrees_increment = kCoarseXRotationDegreesIncrement;
        initialSearchSpace.anglex_rotation_degrees_start = kCoarseXRotationDegreesStart;
        initialSearchSpace.anglex_rotation_degrees_end = kCoarseXRotationDegreesEnd;
        initialSearchSpace.angley_rotation_degrees_increment = kCoarseYRotationDegreesIncrement;
        initialSearchSpace.angley_rotation_degrees_start = kCoarseYRotationDegreesStart;
        initialSearchSpace.angley_rotation_degrees_end = kCoarseYRotationDegreesEnd;
        initialSearchSpace.anglez_rotation_degrees_increment = kCoarseZRotationDegreesIncrement;
        initialSearchSpace.anglez_rotation_degrees_start = kCoarseZRotationDegreesStart;
        initialSearchSpace.anglez_rotation_degrees_end = kCoarseZRotationDegreesEnd;

        cv::Mat outputCandidateElementsMat;
        std::vector< RotationCandidate> candidates;
        cv::Vec3i output_candidate_elements_mat_size;

        ComputeCandidateAngleImages(ball_image1DimpleEdges, initialSearchSpace, outputCandidateElementsMat, output_candidate_elements_mat_size, candidates, local_ball1);

        // Compare the second (presumably rotated) ball image to different candidate rotations of the first ball image to determine the angular change
        std::vector<std::string> comparison_csv_data;
        int best_candidate_index = CompareCandidateAngleImages(&ball_image2DimpleEdges, &outputCandidateElementsMat, &output_candidate_elements_mat_size, &candidates, comparison_csv_data);
        
        cv::Vec3f rotationResult;

        if (best_candidate_index < 0) {
            LoggingTools::Warning("No best candidate found.");
            return rotationResult;
        }

        bool write_spin_analysis_CSV_files = true;

        GolfSimConfiguration::SetConstant("gs_config.spin_analysis.kWriteSpinAnalysisCsvFiles", write_spin_analysis_CSV_files);
        
        if (write_spin_analysis_CSV_files) {
            // This data export can be used for, say, Excel analysis - CSV format
            std::string csv_fname_coarse = "spin_analysis_coarse.csv";
            ofstream csv_file_coarse(csv_fname_coarse);
            GS_LOG_TRACE_MSG(trace, "Writing CSV spin data to: " + csv_fname_coarse);
            for (auto& element : comparison_csv_data)
            {
                // Don't use logging utility so that we don't have all the timing crap in the output
                csv_file_coarse << element;
            }
            csv_file_coarse.close();
        }

        // See which angle looked best and then iterate more closely near those angles
        RotationCandidate c = candidates[best_candidate_index];

        std::string s = "Best Coarse Initial Rotation Candidate was #" + std::to_string(best_candidate_index) + " - Rot: (" + std::to_string(c.x_rotation_degrees) + ", " + std::to_string(c.y_rotation_degrees) + ", " + std::to_string(c.z_rotation_degrees) + ") ";
        GS_LOG_MSG(debug, s);

        // Now iterate more closely in the area that looks best
        RotationSearchSpace finalSearchSpace;

        int anglex_window_width = (int)std::round(ceil(initialSearchSpace.anglex_rotation_degrees_increment / 2.));
        int angley_window_width = (int)std::round(ceil(initialSearchSpace.angley_rotation_degrees_increment / 2.));
        int anglez_window_width = (int)std::round(ceil(initialSearchSpace.anglez_rotation_degrees_increment / 2.));


        finalSearchSpace.anglex_rotation_degrees_increment = 1;
        finalSearchSpace.anglex_rotation_degrees_start = c.x_rotation_degrees - anglex_window_width;
        finalSearchSpace.anglex_rotation_degrees_end = c.x_rotation_degrees + anglex_window_width;
        // Probably not worth it to be too fine-grained on the Y axis.
        finalSearchSpace.angley_rotation_degrees_increment = (int) std::round(kCoarseYRotationDegreesIncrement / 2.);
        finalSearchSpace.angley_rotation_degrees_start = c.y_rotation_degrees - angley_window_width;
        finalSearchSpace.angley_rotation_degrees_end = c.y_rotation_degrees + angley_window_width;
        finalSearchSpace.anglez_rotation_degrees_increment = 1;
        finalSearchSpace.anglez_rotation_degrees_start = c.z_rotation_degrees - anglez_window_width;
        finalSearchSpace.anglez_rotation_degrees_end = c.z_rotation_degrees + anglez_window_width;

        cv::Mat finalOutputCandidateElementsMat;
        cv::Vec3i finalOutputCandidateElementsMatSize;
        std::vector< RotationCandidate> finalCandidates;

        // After this, the finalOutputCandidateElementsMat will have X,Y,Z elements with an index into the finalCandidates vector.
        // Each candidate in finalCandidates will have an image, associated X,Y,Z information and a place to put a score
        ComputeCandidateAngleImages(ball_image1DimpleEdges, finalSearchSpace, finalOutputCandidateElementsMat, finalOutputCandidateElementsMatSize, finalCandidates, local_ball1);

        // TBD - change CompareCandidateAngleImages to work directly with the "3D" images
        best_candidate_index = CompareCandidateAngleImages(&ball_image2DimpleEdges, &finalOutputCandidateElementsMat, &finalOutputCandidateElementsMatSize, &finalCandidates, comparison_csv_data);

        // Save all the candidate scores to a CSV file if requested
        if (write_spin_analysis_CSV_files) {

            std::string csv_fname_fine = "spin_analysis_fine.csv";
            ofstream csv_file_fine(csv_fname_fine);
            GS_LOG_TRACE_MSG(trace, "Writing CSV spin data to: " + csv_fname_fine);
            for (auto& element : comparison_csv_data)
            {
                // Don't use logging utility so that we don't have all the timing crap in the output
                csv_file_fine << element;
            }
            csv_file_fine.close();
        }

        // Analyze the fine-grained results
        int best_rot_x = 0;
        int best_rot_y = 0;
        int best_rot_z = 0;

        if (best_candidate_index >= 0) {
            RotationCandidate finalC = finalCandidates[best_candidate_index];
            best_rot_x = finalC.x_rotation_degrees;
            best_rot_y = finalC.y_rotation_degrees;
            best_rot_z = finalC.z_rotation_degrees;

            // TBD - Experiment - are Y and X reversed?  Try it here...
            // best_rot_x = finalC.y_rotation_degrees;
            // best_rot_y = finalC.x_rotation_degrees;

            std::string s = "Best Raw Fine (and final) Rotation Candidate was #" + std::to_string(best_candidate_index) + " - Rot: (" + std::to_string(best_rot_x) + ", " + std::to_string(best_rot_y) + ", " + std::to_string(best_rot_z) + ") ";
            GS_LOG_MSG(debug, s);

            /*** FOR DEBUG ***/
            cv::Mat bestImg3D = finalCandidates[best_candidate_index].img;
            cv::Mat bestImg2D = cv::Mat::zeros(ball_image1DimpleEdges.rows, ball_image1DimpleEdges.cols, ball_image1DimpleEdges.type());
            Unproject3dBallTo2dImage(bestImg3D, bestImg2D, ball2);
            LoggingTools::DebugShowImage("Best Final Rotation Candidate Image", bestImg2D);
        } 
        else {
            LoggingTools::Warning("No best final candidate found.  Returning 0,0,0 spin results.");
            rotationResult = cv::Vec3d(0, 0, 0);
        }

        // The above angular deltas were calculated relative to a coordinate system that is at an angle
        // from the camera to the balls. So...
        // Now translate the spin angles so that the axes are the same as the PiTrac's and Sim's axes, where, 
        // for example, the Z and Y axes are parallel to the ground plane on which PiTrac sits, and the X axis
        // is orthogonal to that plane

        // We negated the Y offset delta before to account for the Sim's rotational scheme, so will undo here.
        // The idea is to determine the angle to the point in space that was between the two balls.
        cv::Vec3f spin_offset_angle;
        spin_offset_angle[0] = angleOffset1[0] + angleOffsetDeltas1Float[0];
        spin_offset_angle[1] = angleOffset1[1] - angleOffsetDeltas1Float[1];

        GS_LOG_TRACE_MSG(trace, "Now normalizing for spin_offset_angle = (" + std::to_string(spin_offset_angle[0]) + ", " + 
                                    std::to_string(spin_offset_angle[1]) + ", " + std::to_string(spin_offset_angle[2]) + ").");

        double spin_offset_angle_radians_X = CvUtils::DegreesToRadians(spin_offset_angle[0]);
        double spin_offset_angle_radians_Y = CvUtils::DegreesToRadians(spin_offset_angle[1]);
        double spin_offset_angle_radians_Z = CvUtils::DegreesToRadians(spin_offset_angle[2]);

        // Peform the normalization to the real-world axes
        int normalized_rot_x = (int)round( (double)best_rot_x * cos(spin_offset_angle_radians_Y) + (double)best_rot_z * sin(spin_offset_angle_radians_Y) );
        int normalized_rot_y = (int)round( (double)best_rot_y * cos(spin_offset_angle_radians_X) - (double)best_rot_z * sin(spin_offset_angle_radians_X) );

        int normalized_rot_z = (int)round((double)best_rot_z * cos(spin_offset_angle_radians_X) * cos(spin_offset_angle_radians_Y));
        normalized_rot_z -= (int)round((double)best_rot_y * sin(spin_offset_angle_radians_X));
        normalized_rot_z -= (int)round((double)best_rot_x * sin(spin_offset_angle_radians_Y));

        rotationResult = cv::Vec3d(normalized_rot_x, normalized_rot_y, normalized_rot_z);

        GS_LOG_TRACE_MSG(trace, "Normalized spin angles (X,Y,Z) = (" + std::to_string(normalized_rot_x) + ", " + std::to_string(normalized_rot_y) + ", " + std::to_string(normalized_rot_z) + ").");
        
        
        // TBD _ DEBUG
        // See how the original image would look if rotated as the GetBallRotation function calculated
        // We will NOT use the normalized rotations, as the UN-normalized rotations will look most correct
        // in the context of the manner they are imaged by the camera.

        cv::Mat resultBball2DImage;

        GetRotatedImage(ball_image1DimpleEdges, local_ball1, cv::Vec3i(best_rot_x, best_rot_y, best_rot_z), resultBball2DImage);


        if (GolfSimOptions::GetCommandLineOptions().artifact_save_level_ != ArtifactSaveLevel::kNoArtifacts && kLogIntermediateSpinImagesToFile) {
            LoggingTools::LogImage("", resultBball2DImage, std::vector < cv::Point >{}, true, "Filtered Ball1_Rotated_By_Best_Angles.png");
        }

        // We want to show apples to apples, so show the normalized images
        cv::Mat test_ball1_image = normalizedOriginalBallImg1.clone();
        GetRotatedImage(normalizedOriginalBallImg1, local_ball1, cv::Vec3i(best_rot_x, best_rot_y, best_rot_z), test_ball1_image);

        // We'll draw a center-dot on the final image here, but we're not going to re-use that image, so it's ok
        cv::Scalar color{ 0, 0, 0 };
        const GsCircle& circle = local_ball1.ball_circle_;
        cv::circle(test_ball1_image, cv::Point((int)local_ball1.x(), (int)local_ball1.y()), (int)circle[2], color, 2 /*thickness*/);
        LoggingTools::DebugShowImage("Final rotated-by-best-angle originalBall1: ", test_ball1_image, center1);


#ifdef __unix__ 
        // Save the final, rotated, normalized ball result image to the webserver shared directory so that the user
        // can compare them to the original normalized images.
        GsUISystem::SaveWebserverImage(GsUISystem::kWebServerResultBallRotatedByBestAngles, test_ball1_image);
#endif

        // Looks like golf folks consider the X (side) spin to be positive if the surface is
        // going from right to left.  So we negate it here.
        rotationResult[0] = -rotationResult[0];

        // Note that we return angles, not angular velocities.  The velocities will
        // be determined later based on the derived ball speed.
        return rotationResult;
    }


    // Returns the index within candidates that has the best comparison.
    // Returns -1 on failure.
    int BallImageProc::CompareCandidateAngleImages(const cv::Mat* target_image,
                                                    const cv::Mat* candidate_elements_mat,
                                                    const cv::Vec3i* candidate_elements_mat_size,
                                                    std::vector<RotationCandidate>* candidates,
                                                    std::vector<std::string>& comparison_csv_data) {

        boost::timer::cpu_timer timer1;

        // Assume candidates is a vector that is already pre-sized and filled with candidate information
        // and that the candidate_elements_mat has x, y, and z bounds that are commensurate with the candidates vector
        int xSize = (*candidate_elements_mat_size)[0];
        int ySize = (*candidate_elements_mat_size)[1];
        int zSize = (*candidate_elements_mat_size)[2];

        int numCandidates = xSize * ySize * zSize;
        std::vector<std::string> comparisonData(numCandidates);


        // Iterate through the matrix of candidates

        ImgComparisonOp::setup(target_image, candidate_elements_mat, candidates, &comparisonData);

        //  Serialized version for debugging
        if (kSerializeOpsForDebug) {
            for (int x = 0; x < xSize; x++) {
                for (int y = 0; y < ySize; y++) {
                    for (int z = 0; z < zSize; z++) {
                        ushort unusedValue = 0;
                        int position[]{ x, y, z };
                        ImgComparisonOp()(unusedValue, position);
                    }
                }
            }
        }
        else {
            (*candidate_elements_mat).forEach<ushort>(ImgComparisonOp());
        }

        // Find the best candidate from the comparison results
        double maxScaledScore = -1.0;
        double maxPixelsExamined = -1.0;
        double maxPixelsMatching = -1.0;
        int maxPixelsExaminedIndex = -1;
        int maxPixelsMatchingIndex = -1;
        int maxScaledScoreIndex = -1;
        int bestScaledScoreRotX = 0;
        int bestScaledScoreRotY = 0;
        int bestScaledScoreRotZ = 0;
        int bestPixelsMatchingRotX = 0;
        int bestPixelsMatchingRotY = 0;
        int bestPixelsMatchingRotZ = 0;

        // Find the best candidate
        // First, figure out what the largest number of pixels examined were.
        // If we later get a good score, but the number of examined pixels were
        // really low, then we might not want to pick that one.
        // OR... just pick the highest number of matching pixels?  Probably not,
        // as a far rotation that had few pixels to begin with, but very high
        // correspondence might be the correct one

        double kSpinLowCountPenaltyPower = 2.0;
        double kSpinLowCountPenaltyScalingFactor = 1000.0;
        double kSpinLowCountDifferenceWeightingFactor = 500.0;

        double low_count_penalty = 0.0;
        double final_scaled_score = 0.0;

        // Find the range of numbers of matching pixels and the total
        // most-available pixels in order to insert that into the mix for
        // a combined score
        for (auto& element : *candidates)
        {
            RotationCandidate c = element;

            if (c.pixels_examined > maxPixelsExamined) {
                maxPixelsExamined = c.pixels_examined;
                maxPixelsExaminedIndex = c.index;
            }

            if (c.pixels_matching > maxPixelsMatching) {
                maxPixelsMatching = c.pixels_matching;
                maxPixelsMatchingIndex = c.index;
                bestPixelsMatchingRotX = c.x_rotation_degrees;
                bestPixelsMatchingRotY = c.y_rotation_degrees;
                bestPixelsMatchingRotZ = c.z_rotation_degrees;
            }
        }

        for (auto& element : *candidates)
        {
            RotationCandidate c = element;

            low_count_penalty = std::pow((maxPixelsExamined - (double)c.pixels_examined) / kSpinLowCountDifferenceWeightingFactor,
                                kSpinLowCountPenaltyPower) / kSpinLowCountPenaltyScalingFactor;
            final_scaled_score = (c.score * 10.) - low_count_penalty;

            if (final_scaled_score > maxScaledScore) {
                maxScaledScore = final_scaled_score;
                maxScaledScoreIndex = c.index;
                bestScaledScoreRotX = c.x_rotation_degrees;
                bestScaledScoreRotY = c.y_rotation_degrees;
                bestScaledScoreRotZ = c.z_rotation_degrees;
            }
        }

        std::string s = "Best Candidate based on number of matching pixels was #" + std::to_string(maxPixelsMatchingIndex) +
                            " - Rot: (" + std::to_string(bestPixelsMatchingRotX) + ", " + 
                            std::to_string(bestPixelsMatchingRotY) + ", " + std::to_string(bestPixelsMatchingRotZ) + ") ";
        // GS_LOG_MSG(debug, s);

        s = "Best Candidate based on its scaled score of (" + std::to_string(maxScaledScore) + ") was # " + std::to_string(maxScaledScoreIndex) +
                            " - Rot: (" + std::to_string(bestScaledScoreRotX) + ", " + 
                            std::to_string(bestScaledScoreRotY) + ", " + std::to_string(bestScaledScoreRotZ) + ") ";
        GS_LOG_MSG(debug, s);

        // Transfer all the csv data to the output variable
        comparison_csv_data = comparisonData;

        timer1.stop();
        boost::timer::cpu_times times = timer1.elapsed();
        std::cout << "CompareCandidateAngleImages: ";
        std::cout << std::fixed << std::setprecision(8)
            << times.wall / 1.0e9 << "s wall, "
            << times.user / 1.0e9 << "s user + "
            << times.system / 1.0e9 << "s system.\n";

        return maxScaledScoreIndex;
    }





    cv::Vec2i BallImageProc::CompareRotationImage(const cv::Mat& img1, const cv::Mat& img2, const int index) {

        CV_Assert((img1.rows == img2.rows && img1.rows == img2.cols));

        // DEBUG - create a binary image showing what pixels are the same between them
        cv::Mat testCorrespondenceImg = cv::Mat::zeros(img1.rows, img1.cols, img1.type());

        // This comparison is currently done serially, but we should be processing
        // multiple such image comparisons in parallel
        long score = 0;
        long totalPixelsExamined = 0;
        for (int x = 0; x < img1.cols; x++) {
            for (int y = 0; y < img1.rows; y++) {
                uchar p1 = img1.at<uchar>(x, y);
                uchar p2 = img2.at<cv::Vec2i>(x, y)[1];

                if (p1 != kPixelIgnoreValue && p2 != kPixelIgnoreValue) {
                    // Both points have values, so we can validly compare them
                    totalPixelsExamined++;

                    if (p1 == p2) {
                        score++;
                        // The test image is already zero'd out, so only set the
                        // pixel to 1 if there is a match
                        testCorrespondenceImg.at<uchar>(x, y) = 255;
                    }
                }
                else
                {
                    testCorrespondenceImg.at<uchar>(x, y) = kPixelIgnoreValue;
                }
            }
        }

        // LoggingTools::DebugShowImage("testCorrespondenceImg #" + std::to_string(index), testCorrespondenceImg);
        // WON'T WORK BECAUSE IMG2 is 3D LoggingTools::DebugShowImage("testCandidateImg #" + std::to_string(index), img2);

        cv::Vec2i result(score, totalPixelsExamined);
        return result;
    }


    cv::Mat BallImageProc::CreateGaborKernel(int ks, double sig, double th, double lm, double gm, double ps) {

        int hks = (ks - 1) / 2;
        double theta = th * CV_PI / 180;
        double psi = ps * CV_PI / 180;
        double del = 2.0 / (ks - 1);
        double lmbd = lm / 100.0;
        double Lambda = lm;
        double sigma = sig / ks;
        cv::Mat kernel(ks, ks, CV_32F);
        double gamma = gm;

        kernel = cv::getGaborKernel(cv::Size(ks, ks), sig, theta, Lambda, gamma, psi, CV_32F);
        return kernel;
    }

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

    cv::Mat BallImageProc::ApplyTestGaborFilter(const cv::Mat& img_f32,
        const int kernel_size, double sig, double lm, double th, double ps, double gm, float binary_threshold,
        int &white_percent  ) {

        cv::Mat dest = cv::Mat::zeros(img_f32.rows, img_f32.cols, img_f32.type());
        cv::Mat accum = cv::Mat::zeros(img_f32.rows, img_f32.cols, img_f32.type());
        cv::Mat kernel;


        // Sweep through a bunch of different angles for the filter in order to pick up features
        // in all directions
        const double thetaIncrement = 11.25; //  5.625; // CURRENT 11.25;  // degrees.  Nominal: 11.25 also works 
        for (double theta = 0; theta <= 360.0; theta += thetaIncrement) {
            kernel = CreateGaborKernel(kernel_size, sig, theta, lm, gm, ps);
            cv::filter2D(img_f32, dest, CV_32F, kernel);

            cv::max(accum, dest, accum);
        }

        cv::Mat accumGray;

        // Convert from the 0.0 to 1.0 range into 0-255
        accum.convertTo(accumGray, CV_8U, 255, 0);

        cv::Mat dimpleEdges = cv::Mat::zeros(accum.rows, accum.cols, accum.type());

        // Threshold the image to either 0 or 255
        const int edgeThresholdLow = (int)std::round(binary_threshold * 10.);
        const int edgeThresholdHigh = 255;
        cv::threshold(accumGray, dimpleEdges, edgeThresholdLow, edgeThresholdHigh, cv::THRESH_BINARY);

        white_percent = (int)std::round(((double)cv::countNonZero(dimpleEdges) * 100.) / ((double)dimpleEdges.rows * dimpleEdges.cols));

        return dimpleEdges;
    }
 
   bool BallImageProc::ComputeCandidateAngleImages(const cv::Mat& base_dimple_image, 
                                                    const RotationSearchSpace& search_space, 
                                                    cv::Mat &outputCandidateElementsMat,
                                                    cv::Vec3i &output_candidate_elements_mat_size, 
                                                    std::vector< RotationCandidate> &output_candidates, 
                                                    const GolfBall& ball) {
        boost::timer::cpu_timer timer1;

        // These are the ranges of angles that we will create candidate images for
        // We probably won't vary the X-axis rotation much if at all.
        // TBD - Consider a coarse pass first, and then use smaller increments over 
        // the best ROI
        int anglex_rotation_degrees_increment = search_space.anglex_rotation_degrees_increment;
        int anglex_rotation_degrees_start = search_space.anglex_rotation_degrees_start;
        int anglex_rotation_degrees_end = search_space.anglex_rotation_degrees_end;
        int angley_rotation_degrees_increment = search_space.angley_rotation_degrees_increment;
        int angley_rotation_degrees_start = search_space.angley_rotation_degrees_start;
        int angley_rotation_degrees_end = search_space.angley_rotation_degrees_end;
        int anglez_rotation_degrees_increment = search_space.anglez_rotation_degrees_increment;
        int anglez_rotation_degrees_start = search_space.anglez_rotation_degrees_start;
        int anglez_rotation_degrees_end = search_space.anglez_rotation_degrees_end;

        // The ball may not be perfectly centered in the middle of the camera's gaze.  To account for that,
        // the system will essentially rotate the ball to the view the camera would have if it were centered.
        // This is done here by shifting the angles that will be simulated by offsets that account for the
        // ball placement
        
        // TBD - Think hard about how we want to apply the angle offset.  For example, we don't want to 
        // "lose" some of the image because of (for example) moving pixels to the front of the ball from behind it,
        // only to then apply the offset and move the ball back where it was before the pixels were lost.

        // CHANGE - we are going to deal with any camera perspective by pre-de-rotating both of the balls
        // so that they can be compared apples to apples.
        /* - TBD - Delete later when we are sure
        int xAngleOffset = ball.angles_camera_ortho_perspective_[0];
        int yAngleOffset = ball.angles_camera_ortho_perspective_[1];
        anglex_rotation_degrees_start += xAngleOffset;
        anglex_rotation_degrees_end += xAngleOffset;

        angley_rotation_degrees_start += yAngleOffset;
        angley_rotation_degrees_end += yAngleOffset;
        */
        /*  REMOVE - The angle rotations are performed elsewhere currently?? */
        int xAngleOffset = 0;
        int yAngleOffset = 0;


        int xSize = (int)std::ceil((anglex_rotation_degrees_end - anglex_rotation_degrees_start) / anglex_rotation_degrees_increment) + 1;
        int ySize = (int)std::ceil((angley_rotation_degrees_end - angley_rotation_degrees_start) / angley_rotation_degrees_increment) + 1;
        int zSize = (int)std::ceil((anglez_rotation_degrees_end - anglez_rotation_degrees_start) / anglez_rotation_degrees_increment) + 1;

        // Let the caller know what size of matrix we are going to return.  OpenCv only gives rows and columns,
        // so we need to handle this ourselves.

        output_candidate_elements_mat_size = cv::Vec3i(xSize, ySize, zSize);

        GS_LOG_TRACE_MSG(trace, "ComputeCandidateAngleImages will compute " + std::to_string(xSize * ySize * zSize) + " images.");

        // Create a new 3D Mat to hold indexes to the results in the vector.  Use a Mat in order to exploit the forEach() function
        int sizes[3] = { xSize, ySize, zSize };
        outputCandidateElementsMat = cv::Mat(3, sizes, CV_16U, cv::Scalar(0));

        short vectorIndex = 0;

        int xIndex = 0;
        int yIndex = 0;
        int zIndex = 0;

        for (int x_rotation_degrees = anglex_rotation_degrees_start, xIndex = 0; x_rotation_degrees <= anglex_rotation_degrees_end; x_rotation_degrees += anglex_rotation_degrees_increment, xIndex++) {
            for (int y_rotation_degrees = angley_rotation_degrees_start, yIndex = 0; y_rotation_degrees <= angley_rotation_degrees_end; y_rotation_degrees += angley_rotation_degrees_increment, yIndex++) {
                for (int z_rotation_degrees = anglez_rotation_degrees_start, zIndex = 0; z_rotation_degrees <= anglez_rotation_degrees_end; z_rotation_degrees += anglez_rotation_degrees_increment, zIndex++) {

                    cv::Mat ball2DImage;
                    // TBD - Instead of this, call the projectTo3D function and then use the resulting
                    // matrix directly in the comparison
                    // GetRotatedImage(base_dimple_image, ball, cv::Vec3i(x_rotation_degrees, y_rotation_degrees, z_rotation_degrees), ball2DImage);

                    // Project the ball out onto a 3D hemisphere at the current x, y, and z-axis rotation
                    cv::Mat ball13DImage = Project2dImageTo3dBall(base_dimple_image, ball, cv::Vec3i(x_rotation_degrees, y_rotation_degrees, z_rotation_degrees));

                    // Save the current image as a possible candidate to compare to later
                    RotationCandidate c;

                    // The angles in the set of images we are building are angles calculated as if the ball was
                    // centered in the camera's image
                    c.index = vectorIndex;
                    c.img = ball13DImage;
                    c.x_rotation_degrees = x_rotation_degrees - xAngleOffset;
                    c.y_rotation_degrees = y_rotation_degrees - yAngleOffset;
                    c.z_rotation_degrees = z_rotation_degrees;
                    c.score = 0.0;

                    // For now, just throw all of the candidates into a big vector indexed by the entries in the matrix
                    output_candidates.push_back(c);
                    outputCandidateElementsMat.at<ushort>(xIndex, yIndex, zIndex) = vectorIndex;

                    vectorIndex++;
                    
                    // Just for debug for small runs - probably too much information
                    /* std::string s = "ComputeCandidateAngleImages - Rotation Candidate: Idx: " + std::to_string(c.index) +
                        " Rot: (" + std::to_string(c.x_rotation_degrees) + ", " + std::to_string(c.y_rotation_degrees) + ", " + std::to_string(c.z_rotation_degrees) + ") ";
                    GS_LOG_MSG(debug, s);
                    */

                    // FOR DEBUG
                    /*
                    cv::Mat outputGrayImg = cv::Mat::zeros(base_dimple_image.rows, base_dimple_image.cols, base_dimple_image.type());
                    Unproject3dBallTo2dImage(ball13DImage, outputGrayImg, ball);
                    LoggingTools::DebugShowImage("Candidate Image at Rot: (" + std::to_string(c.x_rotation_degrees) + ", " + std::to_string(c.y_rotation_degrees) + ", " + std::to_string(c.z_rotation_degrees) + "): ", outputGrayImg);
                    */
                }
            }
        }

        timer1.stop();
        boost::timer::cpu_times times = timer1.elapsed();
        std::cout << "ComputeCandidateAngleImages Time: " << std::fixed << std::setprecision(8)
            << times.wall / 1.0e9 << "s wall, "
            << times.user / 1.0e9 << "s user + "
            << times.system / 1.0e9 << "s system.\n";

        return true;
    }


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





    // Positive X-axis angles rotate so that the ball appears to go from left to right
    // positive Y-axis angles move the ball from the top to the bottom
    // positive Z-Axis angles are counter-clockwise looking down the positive z-axis
    // The image_gray input Mat is expected to have pixels with only 0, 255, or kPixelIgnoreValue
    cv::Mat BallImageProc::Project2dImageTo3dBall(const cv::Mat& image_gray, const GolfBall& ball, const cv::Vec3i& rotation_angles_degrees) {

        // Create a new 3D Mat to hold the results
        int sizes[2] = { image_gray.rows, image_gray.cols };  // , image_gray.rows };
        // It's possible that due to rotations, some of the 3D image might have "holes" where
        // the pixel was not set to a value.  Make sure anything we don't set is ignored.
        cv::Mat projectedImg = cv::Mat(2, sizes, CV_32SC2, cv::Scalar(0, kPixelIgnoreValue));
        // TBD - hack to pass the 3D image size to the call-back function
        // Kind of a hack, because a 3D Mat won't usually have these values set.  TBD
        projectedImg.rows = image_gray.rows;
        projectedImg.cols = image_gray.cols;

        // Setup the global structures we need before we do the parallelized callback to process
        // the 2D image
        projectionOp::setup(&ball, 
                            projectedImg, 
                            -(float)CvUtils::DegreesToRadians((double)rotation_angles_degrees[0]),  /* Negative due to rotation in X axis being backward */
                            (float)CvUtils::DegreesToRadians((double)rotation_angles_degrees[1]),
                            (float)CvUtils::DegreesToRadians((double)rotation_angles_degrees[2])  );

        if (kSerializeOpsForDebug) {
            /*  Serialized version for debugging - use the parallel stuff below for release */
            for (int x = 0; x < image_gray.cols; x++) {
                for (int y = 0; y < image_gray.rows; y++) {
                    int position[]{ x, y };
                    uchar pixel = image_gray.at<uchar>(x, y);

                    // FOR DEBUG ONLY

                    // TBD - Translate x and y into a new coordinate system that has the origin
                    // at the center of the ball.
                    if (ball.PointIsInsideBall(x, y) && pixel == kPixelIgnoreValue) {
                        GS_LOG_TRACE_MSG(trace, "Project2dImageTo3dBall found ignore pixel within ball at (" + std::to_string(x) + ", " + std::to_string(y) + ").");
                    }


                    projectionOp()(pixel, position);
                }
            }
        }
        else {
            // Parallel execution with function object.
            image_gray.forEach<uchar>(projectionOp());
        }

        return projectedImg;
    }

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

}
