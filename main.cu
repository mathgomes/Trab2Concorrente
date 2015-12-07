#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <cstdio>

using namespace std;
using namespace cv;

//Tamanho da área de cada pedaço a ser feita a filtragem na imagem.
#define SQUARE_AREA 25

//Número de threads a serem utilizadas.
#define NUMTHREAD 1024

__device__ unsigned char smooth(unsigned char* pixel, int i, int j, int w, int h);
__global__ void filtro(unsigned char *in, unsigned char *out, int w, int h);

int main( int argc, char** argv) {

	int tam, tamLargura, tamAltura, blocoX , blocoY, i; 
  double tempo;
  Mat src, dst[3], finImg;
  clock_t itime, ftime;
	unsigned char *dados_in, *dados_out;

  int imgType = atoi(argv[2]);
  src = imread( argv[1], imgType );
	
	//obtém o tamanho da img de entrada.
	tamAltura = src.rows;
	tamLargura = src.cols;
	
	// Aloca vetores para separar a imagem.
	tam = tamLargura * tamAltura * sizeof(unsigned char);
	
	cudaMalloc((void**)&dados_in, tam);
	cudaMalloc((void**)&dados_out, tam);

	// Npumero de blocos para cada dimensão
	blocoX = (int)ceil((double) tamLargura/(double)NUMTHREAD);
	blocoY = tamAltura;

	// define o número de blocos e threads.
	dim3 Blocos(blocoX, blocoY);
	dim3 threadBloco(NUMTHREAD);
	
  itime = clock();
	
  /// grayScale image section
  if( imgType == 0) {
	
    //Passa o filtro no único canal da img em GRAY
		cudaMemcpy(dados_in, (unsigned char*) src.data , tam,  cudaMemcpyHostToDevice);
		filtro<<<Blocos, threadBloco>>>(dados_in, dados_out, tamLargura, tamAltura);
		cudaMemcpy((unsigned char*) src.data , dados_out, tam, cudaMemcpyDeviceToHost);
		imwrite("novaImg.jpg", src);
		
  }else{

	    /// Split the image in channels
	    split(src,dst);
	    
	    /// Apply medianBlur in each channel
	    for(int i=0;i<3;++i){
	    
			cudaMemcpy(dados_in, (unsigned char*) dst[i].data , tam,  cudaMemcpyHostToDevice);
			filtro<<<Blocos, threadBloco>>>(dados_in, dados_out, tamLargura, tamAltura);
			cudaMemcpy((unsigned char*) dst[i].data , dados_out, tam, cudaMemcpyDeviceToHost);
			
	    }

	    /// Push the channels into the Mat vector
	    vector<Mat> rgb;
	    rgb.push_back(dst[0]); //blue
	    rgb.push_back(dst[1]); //green
	    rgb.push_back(dst[2]); //red

	    /// Merge the three channels
	    merge(rgb, finImg);

	    imwrite("novaImg.jpg", finImg);
  }
    
  ftime = clock();
  tempo = (ftime-itime) / (CLOCKS_PER_SEC * 1.0);
  printf("\nTempo : %lf\n",tempo);
	
  cudaFree(dados_in);
	cudaFree(dados_out);
	
  return 0;
}

//Método Smooth para processamento de imagem.
__device__ unsigned char smooth(unsigned char* pixel, int i, int j, int w, int h){
	
	int l, k;
	int sum;
	int raio = 5/2;
	
	sum = 0;
	for(l = i - raio; l <= i + raio; l++) {
		for(k = j - raio; k <= j + raio; k++) {
			if(l >= 0 && k >= 0 && l < h && k < w) {
				sum += pixel[l*w + k];
			}
		}
	}

	return sum/SQUARE_AREA;
}


//método para obter o pixel para a filtragem.
__global__ void filtro(unsigned char *in, unsigned char *out, int w, int h) {
	
	int i, j;

	i = blockIdx.y;
	j = blockIdx.x*blockDim.x + threadIdx.x;

	out[i*w+j] = smooth(in, i, j, w, h);
}
