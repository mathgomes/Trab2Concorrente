
/**
 * file Smoothing.cpp
 * brief Sample code for simple filters
 * author OpenCV team
 */
#include <iostream>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat src; Mat dst[3]; Mat finImg;
char window_name[] = "Smoothing Demo";

int display_dst( int delay );
int display_finImg(int delay);

int medianBLur();

/**
 * function main
 */
int main( void )
{

  /// Load the source image
  src = imread( "../data/baboon.jpg", 1 );

   //src=imread("../data/baboon.jpg");
   split(src,dst);  // planes[2] is the red channel


  //dst = src.clone();
  if( display_dst( DELAY_CAPTION) != 0 ) { return 0; }
   
    // Showing Red Channel
    // G and B channels are kept as zero matrix for visual perception
    vector<Mat> channels;
    channels.push_back(dst[0]);
    channels.push_back(dst[1]);
    channels.push_back(dst[2]);
 	
    /// Merge the three channels
    merge(channels, finImg);

  if( display_finImg( DELAY_CAPTION) != 0 ) { return 0; }

  waitKey(0);

  return 0;
}


int display_finImg(int delay){
  imshow( window_name, finImg);
  int c = waitKey ( delay );
  if( c >= 0 ) { return -1; }	
  return 0;
}

/**
 * @function display_dst
 */
int display_dst( int delay )
{
  int i;

  for(i=0;i<3;++i){ 
 	imshow( window_name, dst[i] );
	int c = waitKey ( delay );
  	if( c >= 0 ) { return -1; }
  }  
  return 0;
  
}
