#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "metrics.h"
#define WIDTH 257

using namespace cv;
using namespace std;

void vis_img(const Mat &img);

void binarize_gt(const Mat &gray);
void binarize_mean_thresh(const Mat &gray, const Mat &out);
float binarize_best_fbeta(const Mat &pred, const Mat &gt, float fprs[WIDTH], float tprs[WIDTH], float precisions[WIDTH], float recalls[WIDTH], float fbetas[WIDTH]);

int main(int argc, char **argv)
{
    // 0: thresh=2*mean_pixel_val, 1: max_fbeta{thresh=0..255}, 2: ?
    int binarize_state = 1;
    // TODO: checl input for right amount of parameters and set binarize state
  //  if ( argc < 4 )
  //  {
  //      printf("usage: ./Metrics <PRED_PATH> <GT_Path> \n");
  //      return -1;
  //  } 
  // //  set binarize method

  // if (*argv[3] == '0')
  // {
  //   // printf("Binarize according to 2*mean.\n");
  //   binarize_state = 0;
  // }
  // else if (*argv[3] == '1')
  // {
  //   // printf("Binarize according to max f_beta.\n");
  //   binarize_state = 1;
  // }else
  // {
  //   printf("No known method chosen!\n");
  //   return 1;
  // }
  
  

    
  // cout << argv[1] << " " << argv[2] << endl;
    // Mat img = imread("/home/schorschi/Documents/SS19/FleckDetect/FleckDataSet/images_train/original/Versuch_Fleck_Probe1_Bot_1_01_20171113_124345.bmp", 0);
    // vis_img(img);
    // equalizeHist(img, img);
    // vis_img(img);

    // Mat gt = imread("/home/schorschi/Documents/SS19/FleckDetect/MSRA-B/gt/0_18_18160.png", 0);
    // Mat pred = imread("/home/schorschi/Documents/SS19/FleckDetect/Results/msra_adam_preds/0_18_18160_pred.png", 0);
    Mat pred = imread(argv[1], 0);
    Mat gt = imread(argv[2], 0);
    float tprs[WIDTH];
    float fprs[WIDTH];
    float precisions[WIDTH];
    float recalls[WIDTH];
    float fbetas[WIDTH];

    float mae = meanAbsoluteError(pred, gt);
    if (binarize_state == 0)
    {
      // binarize_mean_thresh(gt, gt);
      binarize_gt(gt);
      binarize_mean_thresh(pred, pred);
    
  }else if (binarize_state == 1)
  {
    // binarize_best_fbeta(gt, gt);
    // binarize_best_fbeta(pred, pred);
    binarize_best_fbeta(pred, gt, fprs, tprs, precisions, recalls, fbetas);
  }
    
    
  // cout << fprs << endl;
  // Write output in csv format 
  // cout << precision(pred, gt) << "," << recall(pred, gt) << "," << truePositiveRate(pred, gt) << "," << falsePositiveRate(pred, gt)<< "," << fBeta(pred, gt) << "," << mae;

  cout << mae << ',';

  for (int j = 0; j < WIDTH; j++)
  {
    cout << precisions[j] << ',';
  }
  for (int j = 0; j < WIDTH; j++)
  {
    cout << recalls[j] << ',';
  }
  // cout << recalls[255] << endl;
  for (int j = 0; j < WIDTH; j++)
  {
    cout << fbetas[j] << ',';
  }
  for (int j = 0; j < WIDTH; j++)
  {
    cout << fprs[j] << ',';
  }
  for (int j = 0; j < WIDTH-1; j++)
  {
    cout << tprs[j] << ',';
  }
  cout << tprs[WIDTH-1] << endl;
  return 0;
}

/* 
Display image.
@param img input image
 */
void vis_img(const Mat &img) 
{
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", img);
  waitKey(0);
}


void binarize_gt(const Mat &gray)
{
  threshold(gray, gray, 1, 255, THRESH_BINARY);
}

/* 
Binarize image according to 2*mean pixelvalue.
@param gray input inmage in grayscale
@param out binarized image
 */
void binarize_mean_thresh(const Mat &gray, const Mat &out)
{
  Mat out2 = Mat::zeros(gray.rows, gray.cols, CV_8UC1);
  Mat out3 = Mat::zeros(gray.rows, gray.cols, CV_8UC1);
  reduce(gray, out2, 0, REDUCE_AVG, -1);
  reduce(out2, out3, 1, REDUCE_AVG, -1);
  // cout << out2 << endl;
  // cout << out3.at<double>(0,0) << endl;
  int thresh = 2* out3.at<uchar>(0);
  // cout << thresh << endl;

  threshold(gray, out, thresh, 255, THRESH_BINARY);


}

/* 
Calculate fbeta for every threshohld 0-254 and choose the binarization with best value.
As final result both the prediction and ground truth will be binarized.
@param pred prediction  
@param gt Ground truth
 */
float binarize_best_fbeta(const Mat &pred, const Mat &gt, float fprs[WIDTH], float tprs[WIDTH], float precisions[WIDTH], float recalls[WIDTH], float fbetas[WIDTH])
{
  // Mat tmp = pred.clone();
  float max_fbeta = 0;
  int best_thresh = 0;
  // binarize_gt(gt);
  // vis_img(gt);
  for (int i = -1; i <= 255; i++)
  {
    const Mat tmp(pred.size(), CV_8U);
    threshold(pred, tmp, i, 255, THRESH_BINARY);
    fprs[i+1] = falsePositiveRate(tmp, gt);
    tprs[i+1] = truePositiveRate(tmp, gt);
    precisions[i+1] = precision(tmp, gt);
    recalls[i+1] = recall(tmp, gt);
    // vis_img(tmp);
    float fbeta_tmp = fBeta(precisions[i+1], recalls[i+1]);
    fbetas[i+1] = fbeta_tmp;
    // vis_img(tmp);
    // printf("Fbeta: %f\n", fbeta_tmp);
    if (fbeta_tmp > max_fbeta)
    {
      max_fbeta = fbeta_tmp;
      best_thresh = i;
    }
  }
  threshold(pred, pred, best_thresh, 255, THRESH_BINARY);
  return max_fbeta;
}
