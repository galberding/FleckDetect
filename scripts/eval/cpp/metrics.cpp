#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "metrics.h"

using namespace cv;
using namespace std;

float precision(const Mat &prediction, const Mat &groundTruth) {
    // computer the number of pixels predictioned as salient
    const float predictionCount = sum(prediction)[0] / 255.0;
    // if nothing is marked as salient the result depends on the ground truth
    if (predictionCount == 0.0) {
        /*
         * This is an edge case. See:
         *   https://stats.stackexchange.com/questions/8025/what-are-correct-values-for-precision-and-recall-when-the-denominators-equal-0
         *   https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
         *
         * They suggest to give a score of 1.0 if there are no false positives.
         * Instead this implementation returns the false negatives to have a relative score of how many pixels have been missed.
         */
        return 1.0 - (sum(groundTruth)[0] / (255.0 * groundTruth.cols * groundTruth.rows));
    }

    // compute the intersection of the prediction and the ground truth
    const Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, groundTruth, intersection);
    // count the number of pixels predictioned as salient and marked as salient in ground truth
    const float intersectionCount = sum(intersection)[0] / 255.0;
    // return precision
    return intersectionCount / predictionCount;
}

float recall(const Mat &prediction, const Mat &groundTruth) {
    // computer the number of pixels predictioned as salient
    const float groundTruthCount = sum(groundTruth)[0] / 255.0;
    // if nothing is marked as salient the result depends on the ground truth
    if (groundTruthCount == 0.0) {
        /*
         * This is an edge case. See:
         *   https://stats.stackexchange.com/questions/8025/what-are-correct-values-for-precision-and-recall-when-the-denominators-equal-0
         *   https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
         *
         * They suggest to give a score of 1.0 if there are no false negatives.
         * Instead this implementation returns the false positives to have a relative score of how many pixels are incorrectly predicted.
         */
        return sum(prediction)[0] / (255.0 * prediction.cols * prediction.rows);
    }

    // compute the intersection of the prediction and the ground truth
    const Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, groundTruth, intersection);
    // count the number of pixels predictioned as salient and marked as salient in ground truth
    const float intersectionCount = sum(intersection)[0] / 255.0;
    // return precision
    return intersectionCount / groundTruthCount;
}

float truePositiveRate(const Mat &prediction, const Mat &groundTruth) {
    return recall(prediction, groundTruth);
}

float falsePositiveRate(const Mat &prediction, const Mat &groundTruth) {
    const Mat notGroundTruth(prediction.size(), CV_8U);
    bitwise_not(groundTruth, notGroundTruth);
    // computer the number of pixels predictioned as salient
    const float notGroundTruthCount = sum(notGroundTruth)[0] / 255.0;
    // this indicates an erroneous grount truth because if everything is salient than really nothing is
    if (notGroundTruthCount == 0.0) {
        throw std::invalid_argument("Ground truth is completely marked as salient!");
    }

    // compute the intersection of the prediction and the ground truth
    const Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, notGroundTruth, intersection);
    // count the number of pixels predictioned as salient and marked as salient in ground truth
    const float intersectionCount = sum(intersection)[0] / 255.0;
    // return precision
    return intersectionCount / notGroundTruthCount;
}

float fBeta(const Mat &prediction, const Mat &groundTruth, float betaSquared) {
    return fBeta(precision(prediction, groundTruth), recall(prediction, groundTruth), betaSquared);
}

float fBeta(float precision, float recall, float betaSquared) {
    float numerator = (1 + betaSquared) * precision * recall;
    float denominator = (betaSquared) * precision + recall;
    if (numerator == 0 || denominator == 0)
    {
        return 0;
    }else{
        return numerator / denominator;
    }
}

float intersectionOverUnion(const Mat &prediction, const Mat &groundTruth) {
    // compute the intersection of the prediction and the ground truth
    const Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, groundTruth, intersection);
    // count the number of pixels predictioned as salient and marked as salient in ground truth
    const float intersectionCount = sum(intersection)[0] / 255.0;

    // compute the union of the prediction and the ground truth
    const Mat _union(prediction.size(), CV_8U);
    bitwise_or(prediction, groundTruth, _union);
    // count the number of pixels predictioned as salient and marked as salient in ground truth
    const float unionCount = sum(_union)[0] / 255.0;
    //
    // return precision
    return intersectionCount / unionCount;
}

float meanAbsoluteError(const Mat &saliencyMap, const Mat &groundTruth){
    const Mat absoluteDifference(saliencyMap.size(), CV_8U);
    absdiff(saliencyMap, groundTruth, absoluteDifference);
    return sum(absoluteDifference)[0] / float(absoluteDifference.size().height * absoluteDifference.size().width * 255.0);
}

/**
 * Compute the contour of an image region detected with openCV connectedComponentsWithStats method.
 *
 * @param image the image on which the connected components were detected
 * @param labels the image where each pixel has a region label as created by connectedComponentsWithStats
 * @param componentLabel the label of the component of which the contour is computed
 * @param componentStats array generated by the openCV method
 * @return a vector of points describing the contour of the region
 */
vector<Point> computeRegionContour(const Mat &image, const Mat &labels, const int componentLabel, const Mat &componentStats) {
    // extract region of interest as bounding rect of connected component
    const int x = componentStats.at<int>(componentLabel, CC_STAT_LEFT);
    const int y = componentStats.at<int>(componentLabel, CC_STAT_TOP);
    const Rect roi(
            componentStats.at<int>(componentLabel, CC_STAT_LEFT),
            componentStats.at<int>(componentLabel, CC_STAT_TOP),
            componentStats.at<int>(componentLabel, CC_STAT_WIDTH),
            componentStats.at<int>(componentLabel, CC_STAT_HEIGHT)
        );

    // create copy with border of zeros of ROI to extract the contour from
    // there are two reasons for this described in https://github.com/opencv/opencv/issues/4374 
    //  * a copy has to be created because the findContours method modifies the original image
    //  * the border has to be created because the indices are moved at borders
    Mat roiWithBorder;
    copyMakeBorder(image(roi), roiWithBorder, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    // create an offset point so that the computed contour is in original image coordinates 
    const Point offset(
            componentStats.at<int>(componentLabel, CC_STAT_LEFT) - 1,
            componentStats.at<int>(componentLabel, CC_STAT_TOP) - 1
        );
    // create contours vector
    vector<vector<Point>> contours;
    // find contours (only external because there should only be one)
    // TODO: Raises error
    // findContours(roiWithBorder, contours, CV_EXTERN_C, CHAIN_APPROX_NONE, offset);

    // return the contour which has the given label
    for (const vector<Point>& contour : contours) {
        if (labels.at<int>(contour[0]) == componentLabel) {
            return contour;
        }
    } 
    // fail because no contour was found
    assert(false);
    return contours[0];
}

float regionDetectionRate(const Mat &prediction, const Mat &groundTruth, const float objectCoveredThreshold, const float predictionCoverageThreshold, const float distanceThreshold, const int minAreaSize) {
    // compute connected components with 8 way connectivity
    const int connectivity = 8;
    // save labels as mat with integer types
    const int labelType = CV_32S;

    // compute the intersection between ground truth and prediction
    const Mat intersection(prediction.size(), CV_8U);
    bitwise_and(prediction, groundTruth, intersection);

    // compute connected components of intersection
    Mat intersectionLabels, intersectionStats, intersectionCentroids;
    int intersectionLabelCount =
        connectedComponentsWithStats(intersection, intersectionLabels, intersectionStats, intersectionCentroids, connectivity, labelType);

    // compute connected components of ground truth
    Mat groundTruthLabels, groundTruthStats, groundTruthCentroids;
    int groundTruthLabelCount =
        connectedComponentsWithStats(groundTruth, groundTruthLabels, groundTruthStats, groundTruthCentroids, connectivity, labelType);

    // compute connected components of prediction
    Mat predictionLabels, predictionStats, predictionCentroids;
    int predictionLabelCount =
        connectedComponentsWithStats(prediction, predictionLabels, predictionStats, predictionCentroids, connectivity, labelType);

    // save for each prediction the percentage of ground truth it covers
    float predictionCoverage[predictionLabelCount] = {0.0};
    // save for each region in the ground truth which prediction covers how much
    map<int, map<int, float>> coveredByPredictionMap;

    // iterate over all regions in the intersection to fill the array and the map created above
    // note that this and all following loops start iteration at 1 because 0 is the background label
    for (int intersectionLabel = 1; intersectionLabel < intersectionLabelCount; intersectionLabel++) {
        // create a point at the top left of the bounding rectangle of the region in the intersection
        Point pointWithLabel = Point(
                intersectionStats.at<int>(intersectionLabel, CC_STAT_LEFT),
                intersectionStats.at<int>(intersectionLabel, CC_STAT_TOP)
                );
        // search the border of the bounding rect for a point on the region
        // this has to work because the bounding rect created is inclusive
        while (intersectionLabels.at<int>(pointWithLabel) != intersectionLabel) {
            pointWithLabel.x++;
        }

        // because the intersection has a label at that point, the prediction and the ground truth also
        // have to have labels there
        int groundTruthLabel = groundTruthLabels.at<int>(pointWithLabel);

        // skip regions which are too small
        if (groundTruthStats.at<int>(groundTruthLabel, CC_STAT_AREA) < minAreaSize) {
            continue;
        }

        int predictionLabel = predictionLabels.at<int>(pointWithLabel);
        assert(groundTruthLabel != 0 && predictionLabel != 0); // 0 is the background label which should not be retrieved here

        // add the percentage covered of the ground truth region to the map
        coveredByPredictionMap[predictionLabel][groundTruthLabel] = 
            intersectionStats.at<int>(intersectionLabel, CC_STAT_AREA) / (float) groundTruthStats.at<int>(groundTruthLabel, CC_STAT_AREA);
        // add the percentage of coverage of the prediction to the array
        predictionCoverage[predictionLabel] +=
            intersectionStats.at<int>(intersectionLabel, CC_STAT_AREA) / (float) predictionStats.at<int>(predictionLabel, CC_STAT_AREA); 
    }

    // after the creation of the map and array, it is time to validate the two "or" criteria of the metric
    // and to save how much of each salient region is covered
    // save for each region in the ground truth how much of it is covered by the prediction
    float groundTruthRegionCoverage[groundTruthLabelCount] = { 0.0 };

    // iterate over all predicted regions
    for (int predictionLabel = 1; predictionLabel < predictionLabelCount; predictionLabel++) {
        if (coveredByPredictionMap.find(predictionLabel) == coveredByPredictionMap.end()) {
            continue;
        }

        const map<int, float> coverageMap = coveredByPredictionMap[predictionLabel];
        if (predictionCoverage[predictionLabel] >= predictionCoverageThreshold) {
            // if this prediction covers over a certain treshold of salient objects, add its coverage to the according regions
            for (auto const& entry : coverageMap) {
                groundTruthRegionCoverage[entry.first] += entry.second;
            }
        } else {
            // else test if the maximum of minimal contour distances is lower than a threshold

            // compute the contour of the prediction region
            const vector<Point> predictionContour = computeRegionContour(prediction, predictionLabels, predictionLabel, predictionStats);

            // iterate over all ground truth regions it intersects
            for (auto const& entry : coverageMap) {
                int groundTruthLabel = entry.first;
                // compute the contour of the ground truth region
                const vector<Point> groundTruthContour = computeRegionContour(groundTruth, groundTruthLabels, groundTruthLabel, groundTruthStats);

                // save the max distance
                float maxDistance = numeric_limits<float>::min();
                // iterate over all prediction contour points
                for (auto const& predictionPoint : predictionContour) {
                    // compute the signed distance
                    float distance = pointPolygonTest(groundTruthContour, predictionPoint, true);

                    // skip if the distance is greater than 0 which means that the prediction point is inside the ground truth region
                    if (distance >= 0) {
                        continue;
                    }

                    // compare with the abdolue value because points outside have a negative distance
                    distance = abs(distance);
                    if (distance > maxDistance) {
                        maxDistance = distance;
                    }
                }

                // if the distance is smaller than a threshold for this ground truth region add its coverage
                if (maxDistance <= distanceThreshold) {
                    groundTruthRegionCoverage[groundTruthLabel] += entry.second;
                }
            }
        }
    }

    // save how many areas are larger than minArea size
    int areaCount = groundTruthLabelCount - 1;

    // compute how many ground truth regions are covered above a threshold
    int regionsDetected = 0;
    for (int groundTruthLabel = 1; groundTruthLabel < groundTruthLabelCount; groundTruthLabel++) {
        if (groundTruthStats.at<int>(groundTruthLabel, CC_STAT_AREA) < minAreaSize) {
            areaCount--;
            continue;
        }
        if (groundTruthRegionCoverage[groundTruthLabel] >= objectCoveredThreshold) {
            regionsDetected++;
        }
    }

    // compute the percentage of objects covered
    return regionsDetected / (float) areaCount;
}
