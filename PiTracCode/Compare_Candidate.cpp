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