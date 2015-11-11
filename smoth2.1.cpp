#include <iostream>
#include <vector>
#include <string>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

#define SQUARE_AREA 25

Mat mediaBlur(Mat& img);
int display_finImg(int delay, String&, Mat&);


int main( int argc, char** argv) {

    /// General use variables
    int DELAY_CAPTION = 1500, top = 2, bottom = 2, left = 2, right = 2, borderType = BORDER_CONSTANT;
    String window_name = "Smoothing ";
    Mat src, dst[3], finImg;

    ///Values that will be inserted in image border( solves border pixels problem )
    Scalar value( 0, 0, 0);

    /// Load the source image
    int *imgType = (int*)argv[2];
    src = imread( argv[1], *imgType );

    /// grayScale image section
    if( *imgType == 0) {
        finImg = src.clone();
        copyMakeBorder(finImg,finImg,top,bottom,left,right,borderType, value);
        finImg = mediaBlur(finImg);
        finImg = finImg.colRange(3, (finImg.cols-3));
        finImg = finImg.rowRange(3, (finImg.rows-3));
        if( display_finImg( DELAY_CAPTION,window_name,finImg) != 0 ) { return 0; }
        waitKey(0);
        return 0;
    }

    /// Split the image in channels
    split(src,dst);

    /// Apply medianBlur in each channel
    for(int i=0;i<3;++i){

        //insere a borda de tamanhos especificados
        copyMakeBorder( dst[i], dst[i], top, bottom, left, right, borderType, value );
        dst[i] = mediaBlur(dst[i]);

        //as duas linhas abaixo devolvem a matriz sem as bordas.
        dst[i] = dst[i].colRange(3, (dst[i].cols-3));
        dst[i] = dst[i].rowRange(3, (dst[i].rows-3));
    }


    /// Push the channels into the Mat vector
    vector<Mat> rgb;
    rgb.push_back(dst[0]); //blue
    rgb.push_back(dst[1]); //green
    rgb.push_back(dst[2]); //red


    /// Merge the three channels
    merge(rgb, finImg);

    if( display_finImg( DELAY_CAPTION,window_name,finImg) != 0 ) { return 0; }

    waitKey(0);

    return 0;
}


Mat mediaBlur(Mat &channel){

    int rowSize = channel.rows, colSize = channel.cols, sum;
    double media;

    for(int i = 2; i < (rowSize-2); ++i){

        for(int j = 2; j < (colSize-2); ++j){
            sum=0;
            for( int l = i - 2; l <= i + 2; l++) {

                for( int c = j - 2; c <= j + 2; c++) {
                    sum += channel.at<uchar>(l,c);

                }
            }
            media = sum/SQUARE_AREA;
            channel.at<uchar>(i,j) = media;
        }
    }
    return channel;
}


int display_finImg(int delay, String &window_name, Mat &finImg){

    imshow( window_name, finImg);
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
}
