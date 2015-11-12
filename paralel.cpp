#include <iostream>
#include <vector>
#include <string>

#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <omp.h>
#include <cstdio>

using namespace std;
using namespace cv;

#define SQUARE_AREA 25
//o processador é um quad core
#define NUMTHREADS 4

Mat src, dst[3], dst2[3], finImg;

void mediaBlurRGB(int inicio, int fim, int indice);
void mediaBlur(int inicio, int fim);
int display_finImg(int delay, String&, Mat&);

int main( int argc, char** argv) {

    /// General use variables
    int DELAY_CAPTION = 1500, top = 2, bottom = 2, left = 2, right = 2, borderType = BORDER_CONSTANT, tamLin, bloco, resto;
    String window_name = "Smoothing ";
    clock_t itime, ftime;

    ///Values that will be inserted in image border( solves border pixels problem )
    Scalar value( 0, 0, 0);

    /// Load the source image
    int imgType = atoi(argv[2]);
    src = imread( argv[1], imgType );

    //copyMakeBorder(src,src, top, bottom,left, right, borderType, value);
    finImg = src.clone();
    
    tamLin = finImg.rows;
	
    //divide a img em partes iguais para paralelizar no openmp
    bloco = tamLin/NUMTHREADS;
    resto = tamLin%NUMTHREADS;

    copyMakeBorder( finImg, finImg, top, bottom, left, right, borderType, value );
    itime = clock();
    /// grayScale image section
    if(imgType == 0) {

	//percorrer o loop para cada bloco da imagem definida.
	#pragma omp parallel for
	for(int i=0; i < NUMTHREADS; ++i){
	   if(i == NUMTHREADS-1)
	      mediaBlur(bloco*i, (bloco*(i+1)) - 1);
	   else
	      mediaBlur(bloco*i, (bloco*(i+1)) - 1 + resto);
	}

        finImg = finImg.colRange(3, (finImg.cols-3));
        finImg = finImg.rowRange(3, (finImg.rows-3));
        if( display_finImg( DELAY_CAPTION,window_name,finImg) != 0 ) { return 0; }
        waitKey(0);
    } else {

	    /// Split the image in channels
	    split(src,dst);
	    split(src, dst2);

	    /// Apply medianBlur in each channel
	    for(int j=0;j<3;++j){

		//insere a borda de tamanhos especificados
		copyMakeBorder( dst[j], dst[j], top, bottom, left, right, borderType, value );
		copyMakeBorder( dst2[j], dst2[j], top, bottom, left, right, borderType, value );

		//percorrer o loop para cada bloco da imagem definida.
		#pragma omp parallel for
		for(int i=0; i < NUMTHREADS; ++i){
		   if(i == NUMTHREADS-1)
		      mediaBlurRGB(bloco*i, ((bloco*(i+1)) - 1), j);
		   else
		      mediaBlurRGB((bloco)*i, ((bloco*(i+1)) - 1 + resto), j);
		}

		//as duas linhas abaixo devolvem a matriz sem as bordas.
		dst[j] = dst[j].colRange(3, (dst[j].cols-3));
		dst[j] = dst[j].rowRange(3, (dst[j].rows-3));
	    }


	    /// Push the channels into the Mat vector
	    vector<Mat> rgb;
	    rgb.push_back(dst[0]); //blue
	    rgb.push_back(dst[1]); //green
	    rgb.push_back(dst[2]); //red


	/// Merge the three channels
	merge(rgb, finImg);
   	//if( display_finImg( DELAY_CAPTION,window_name,finImg) != 0 ) { return 0; }
    	//waitKey(0);
    }
    
ftime = clock();
    tempo = (ftime-itime) / (CLOCKS_PER_SEC * 1.0);
    printf("\nTempo : %lf\n",tempo);

    return 0;
}


void mediaBlurRGB(int inicio, int fim, int indice){

    int colSize = finImg.cols, sum, inicioLinha;
    double media;

    if(inicio == 0)
	inicioLinha = inicio + 2;
    else
        inicioLinha = inicio;

    for(int i = inicioLinha; i < fim+2; ++i){

        for(int j = 2; j < (colSize-2); ++j){
            sum=0;
            for( int l = i - 2; l <= i + 2; l++) {

                for( int c = j - 2; c <= j + 2; c++) {
                    sum += dst2[indice].at<uchar>(l,c);

                }
            }
	media = sum/SQUARE_AREA;
        dst[indice].at<uchar>(i,j) = media;
        }
    }
}

void mediaBlur(int inicio, int fim){
    
    int colSize = finImg.cols, sum, inicioLinha;
    double media;
    
    if(inicio == 0)
	inicioLinha = inicio + 2;
    else
        inicioLinha = inicio;

    for(int i = inicioLinha; i < fim+2; ++i){

        for(int j = 2; j < (colSize-2); ++j){
            sum=0;
            for( int l = i - 2; l <= i + 2; l++) {

                for( int c = j - 2; c <= j + 2; c++) {
                    sum += finImg.at<uchar>(l,c);

                }
            }
	media = sum/SQUARE_AREA;
        finImg.at<uchar>(i,j) = media;
        }
    }
}


int display_finImg(int delay, String &window_name, Mat &finImg){

    imshow( window_name, finImg);
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
}
