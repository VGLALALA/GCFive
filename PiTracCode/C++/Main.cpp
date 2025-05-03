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