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

    int DELAY_CAPTION = 1500;
    String window_name = "Smoothing ";

    Mat src, dst[3], finImg;

    /// Load the source image
    int *imgType = (int*)argv[2];
    src = imread( argv[1], *imgType );

    /// Split the image in channels
    split(src,dst);

    /// Apply medianBlur in each channel
    for(int i=0;i<3;++i){
        dst[i] = mediaBlur(dst[i]);
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

    for(int i = 0; i < rowSize; ++i){

        for(int j = 0; j < colSize; ++j){

            sum=0;

	    if((i-2)>=0 && (j-2)>=0){
		sum += channel.at<uchar>(i-2,j-2);
	    }

	    if((i-2)>=0 && (j-1)>=0 && (i-2) < rowSize && (j-1) < colSize){
		sum += channel.at<uchar>(i-2,j-1);
	    }
	    
	    if((i-2)>=0 && (j)>=0 && (i-2) < rowSize && j < colSize){
		sum += channel.at<uchar>(i-2,j);
	    }
	    
            if((i-2)>=0 && (j+1)>=0 && (i-2) < rowSize && (j+1) < colSize){
		sum += channel.at<uchar>(i-2,j+1);
	    }

	    if((i-2)>=0 && (j+2)>=0 && (i-2) < rowSize && (j+2) < colSize){
		sum += channel.at<uchar>(i-2,j+2);
	    }

	    if((i-1)>=0 && (j-2)>=0 && (i-1) < rowSize && (j-2) < colSize){
		sum += channel.at<uchar>(i-1,j-2);
	    }

	    if((i-1)>=0 && (j-1)>=0 && (i-1) < rowSize && (j-1) < colSize){
		sum += channel.at<uchar>(i-1,j-1);
	    }

	    if((i-1)>=0 && (j)>=0 && (i-1) < rowSize && j < colSize){
		sum += channel.at<uchar>(i-1,j);
	    }

	    if((i-1)>=0 && (j+1)>=0 && (i-1) < rowSize && (j+1) < colSize){
		sum += channel.at<uchar>(i-1,j+1);
	    }

	    if((i-1)>=0 && (j+2)>=0 && (i-1) < rowSize && (j+2) < colSize){
		sum += channel.at<uchar>(i-1,j+2);
	    }
	        
	    if((i)>=0 && (j-2)>=0 && i < rowSize && (j-2) < colSize){
		sum += channel.at<uchar>(i,j-2);
	    }

	    if((i)>=0 && (j-1)>=0 && i < rowSize && (j-1) < colSize){
		sum += channel.at<uchar>(i,j-1);
	    }

	    if((i)>=0 && (j+1)>=0 && i < rowSize && (j+1) < colSize){
		sum += channel.at<uchar>(i,j+1);
	    }

	    if((i)>=0 && (j+2)>=0 && i < rowSize && (j+2) < colSize){
		sum += channel.at<uchar>(i,j+2);
	    }

	    if((i+1)>=0 && (j-2)>=0 && (i+1) < rowSize && (j-2) < colSize){
		sum += channel.at<uchar>(i+1,j-2);
	    }

	    if((i+1)>=0 && (j-1)>=0 && (i+1) < rowSize && (j-1) < colSize){
		sum += channel.at<uchar>(i+1,j-1);
	    }

	    if((i+1)>=0 && (j)>=0 && (i+1) < rowSize && j < colSize){
		sum += channel.at<uchar>(i+1,j);
	    }

	    if((i+1)>=0 && (j+1)>=0 && (i+1) < rowSize && (j+1) < colSize){
		sum += channel.at<uchar>(i+1,j+1);
	    }

	    if((i+1)>=0 && (j+2)>=0 && (i+1) < rowSize && (j+2) < colSize){
		sum += channel.at<uchar>(i+1,j+2);
	    }

	    if((i+2)>=0 && (j-2)>=0 && (i+2) < rowSize && (j-2) < colSize){
		sum += channel.at<uchar>(i+2,j-2);
	    }

	    if((i+2)>=0 && (j-1)>=0 && (i+2) < rowSize && (j-1) < colSize){
		sum += channel.at<uchar>(i+2,j-1);
	    }

	    if((i+2)>=0 && (j)>=0 && (i+2) < rowSize && j < colSize){
		sum += channel.at<uchar>(i+2,j);
	    }

	    if((i+2)>=0 && (j+1)>=0 && (i+2) < rowSize && (j+1) < colSize){
		sum += channel.at<uchar>(i+2,j+1);
	    }

	    if((i+2)>=0 && (j+2)>=0 && (i+2) < rowSize && (j+2) < colSize){
		sum += channel.at<uchar>(i+2,j+2);
	    }

		
	    sum += channel.at<uchar>(i,j);
            media = sum/SQUARE_AREA;
            channel.at<uchar>(i,j) = media;

            /*for(int l =(i-2); l < (i+2) ;++l){

                for(int c =(j-2); c < (j+2) ;++c){

                    if(l>=0 && c>=0 && l < rowSize && c < colSize){
                        sum += channel.at<uchar>(l,c);
                    }
                }
            }
            media = sum/SQUARE_AREA;
            channel.at<uchar>(i,j) = media;*/
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
