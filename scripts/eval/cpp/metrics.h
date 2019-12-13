#ifndef EVALUATION_METRICS_H
#define EVALUATION_METRICS_H


#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

/**
 * Compute the precision score for a prediction and a ground truth image.
 * The score will be in the interval [0, 1] and a higher value is better.
 *
 * @param prediction the prediction
 * @param groundTruth the ground truth
 * @return the precision score for the given prediction and the ground truth.
 */
float precision(const Mat &prediction, const Mat &groundTruth);

/**
 * Compute the precision score for a prediction and a ground truth image.
 * The score will be in the interval [0, 1] and a higher value is better.
 *
 * @param prediction the prediction.
 * @param groundTruth the ground truth.
 * @return the precision score for the given prediction and the ground truth.
 */
float recall(const Mat &prediction, const Mat &groundTruth);

/**
 * Compute the true positive rate.
 * This is the percentage of pixels in the ground truth correctly classified.
 * This is equal to the recall.
 *
 * @param prediction the prediction
 * @param groundTruth the ground truth.
 * @return the precision score for the given prediction and the ground truth.
 */
float truePositiveRate(const Mat &prediction, const Mat &groundTruth);

/**
 * Compute the false positive rate.
 * This is the percentage of pixels marked falsely as salient. 
 * Therefore, a lower value is better.
 *
 * @param prediction the prediction
 * @param groundTruth the ground truth.
 * @return the false negative rate.
 */
float falsePositiveRate(const Mat &prediction, const Mat &groundTruth);

/**
 * Compute the f beta measure from a prediction and a ground truth.
 *
 * @param prediction the prediction
 * @param groundTruth the ground truth
 * @param betaSquared the beta squared value which defaults to 0.3
 * @return the value of the f beta measure.
 */
float fBeta(const Mat &prediction, const Mat &groundTruth, float betaSquared=0.3);

/**
 * Compute the f beta measure from recall and precision scores.
 *
 * @param precision the precision score
 * @param recall the recall score
 * @param beta the beta value which defaults to 0.3
 * @return the value of the f beta measure.
 */
float fBeta(float precision, float recall, float beta=0.3);

/**
 * Compute the intersection over union score.
 *
 * @param prediction the prediction
 * @param groundTruth the ground truth
 * @return the intersection over union score
 */
float intersectionOverUnion(const Mat &prediction, const Mat &groundTruth);

/**
 * Compute the region measure as describe in the master thesis of Thomas Krahn.
 *
 * @param prediction the prediction
 * @param groundTruth the ground truth
 * @param predictionCoverageThreshold the threshold of how much of a salient object has to be correctly predicted (default is 0.5)
 * @param predictionCoverageThreshold the threshold of how much of a predicted region can be non salient
 * @param distanceThreshold the threshold of the maximum minimal distance
 * @param minAreaSize the area size in pixels up to which regions from the ground truth are ignored
 * @return the region detection rate
 */
float regionDetectionRate(const Mat &prediction, const Mat &groundTruth, const float objectCoveredThreshold=0.5, const float predictionCoverageThreshold=0.25, const float distanceThreshold=100, const int minAreaSize=100);

/**
 * Compute the mean absolute error.
 *
 * @param saliencyMap the saliency map
 * @param ground truth the ground truth
 * @return the mean absolute error
 */
float meanAbsoluteError(const Mat &saliencyMap, const Mat &groundTruth);

#endif
