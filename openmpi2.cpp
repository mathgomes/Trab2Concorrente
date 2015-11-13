#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#define NTHREADS 4

using namespace cv;
using namespace std;
/*
	mpic++ -o SmoothMPI_MP SmoothMPI_MP.cpp `pkg-config --cflags --libs opencv` -fopenmp
	mpirun -np 4 SmoothMPI_MP image_in type_img image_out
*/

Mat out;
int border = 2;

void aplica_smooth_grayscale(Mat& in, int ini_i, int ini_j, int end_i, int end_j) {
	float average;
	for(int i = ini_i + border; i < end_i - border; i++) {
		for(int j = ini_j + border; j < end_j - border; j++) {
			average = in.at<uchar>(i, j) 
			+ in.at<uchar>(i + 1, j) 
			+ in.at<uchar>(i - 1, j) 
			+ in.at<uchar>(i, j + 1) 
			+ in.at<uchar>(i, j - 1) 
			+ in.at<uchar>(i + 2, j)
			+ in.at<uchar>(i - 2, j)
			+ in.at<uchar>(i, j + 2)
			+ in.at<uchar>(i, j - 2)
			+ in.at<uchar>(i + 1, j + 1) 
			+ in.at<uchar>(i + 1, j - 1) 
			+ in.at<uchar>(i + 1, j - 2) 
			+ in.at<uchar>(i + 1, j + 2)
			+ in.at<uchar>(i - 1, j + 1)
			+ in.at<uchar>(i - 1, j - 1)
			+ in.at<uchar>(i - 1, j + 2)
			+ in.at<uchar>(i - 1, j - 2)
			+ in.at<uchar>(i + 2, j + 1)
			+ in.at<uchar>(i + 2, j - 1)
			+ in.at<uchar>(i + 2, j + 2)
			+ in.at<uchar>(i + 2, j - 2)
			+ in.at<uchar>(i - 2, j + 1)
			+ in.at<uchar>(i - 2, j - 1)
			+ in.at<uchar>(i - 2, j + 2)
			+ in.at<uchar>(i - 2, j - 2);
			average = average/25;	
			out.at<uchar>(i - border, j - border) = (uchar)average;
		}
	}
}
void aplica_smooth_color(Mat &in, int ini_i, int ini_j, int end_i, int end_j) {
	float average;
	for(int i = ini_i + border; i < end_i  - border; i++) {
		for(int j = ini_j + border; j < end_j - border; j++) {
			for(int color = 0; color < 3; color++) {
				average = in.at<cv::Vec3b>(i, j)[color] 
				+ in.at<cv::Vec3b>(i + 1, j)[color] 
				+ in.at<cv::Vec3b>(i - 1, j)[color]
				+ in.at<cv::Vec3b>(i, j + 1)[color] 
				+ in.at<cv::Vec3b>(i, j - 1)[color]
				+ in.at<cv::Vec3b>(i - 2, j)[color] 
				+ in.at<cv::Vec3b>(i + 2, j)[color] 
				+ in.at<cv::Vec3b>(i, j + 2)[color]
				+ in.at<cv::Vec3b>(i, j - 2)[color]
				+ in.at<cv::Vec3b>(i + 1, j + 1)[color]
				+ in.at<cv::Vec3b>(i + 1, j + 2)[color] 
				+ in.at<cv::Vec3b>(i + 1, j - 1)[color] 
				+ in.at<cv::Vec3b>(i + 1, j - 2)[color]
				+ in.at<cv::Vec3b>(i - 1, j - 1)[color] 
				+ in.at<cv::Vec3b>(i - 1, j + 1)[color]
				+ in.at<cv::Vec3b>(i - 1, j + 2)[color]
				+ in.at<cv::Vec3b>(i - 1, j - 2)[color] 
				+ in.at<cv::Vec3b>(i - 2, j - 2)[color] 
				+ in.at<cv::Vec3b>(i - 2, j + 1)[color]
				+ in.at<cv::Vec3b>(i - 2, j - 1)[color]
				+ in.at<cv::Vec3b>(i - 2, j + 2)[color]
				+ in.at<cv::Vec3b>(i + 2, j + 2)[color]
				+ in.at<cv::Vec3b>(i + 2, j + 1)[color] 
				+ in.at<cv::Vec3b>(i + 2, j - 1)[color]
				+ in.at<cv::Vec3b>(i + 2, j - 2)[color];

				average = average/25;
				out.at<cv::Vec3b>(i - border, j - border)[color] = (uchar)average;
			}
		}
	}
}
void smooth(Mat& in,int tipo_img) {

	//copia imagem de entrada
	if(tipo_img == 0) {
		out = Mat(in.size(), CV_8U, 1);
	} else if(tipo_img == 1) {
		out = Mat(in.size(), CV_8UC3, 1);
	}

	//achar a média aritmética e depois atualizar o pixel
	float average;
	omp_set_num_threads(NTHREADS);	

	int ini_i[4],ini_j[4],end_i[4],end_j[4];
	int meio_w, meio_h;
	copyMakeBorder(in, in, border, border, border, border, BORDER_REPLICATE);

	//calcula o meio da imagem
	meio_h = in.size().height/2;
	meio_w = in.size().width/2;
	//calcula o inicio de cada pedaço da imagem
	ini_i[0] = 0; 							ini_j[0] = 0;
	ini_i[1] = 0; 							ini_j[1] = meio_w - border;
	ini_i[2] = meio_h - border; 			ini_j[2] = 0;
	ini_i[3] = meio_h - border; 			ini_j[3] = meio_w - border;
	//calcula o fim de cada pedaço da imagem
	end_i[0] = meio_h + border; 			end_j[0] = meio_w + border;
	end_i[1] = meio_h + border; 			end_j[1] = in.size().width;
	end_i[2] = in.size().height;		 	end_j[2] = meio_w + border;
	end_i[3] = in.size().height;			end_j[3] = in.size().width;

	//chama cada iteração do for junto em paralelo passando como parametro o 
	//inicio e o fim de cada pedaço da imagem qu	e cada thread deve rodar
	#pragma omp parallel for shared(in,out, end_i, end_j, ini_i, ini_j)
		for(int i = 0; i < NTHREADS; i ++) {
			if(tipo_img == 0) {
				aplica_smooth_grayscale(in, ini_i[i], ini_j[i], end_i[i], end_j[i]);
			} else {
				aplica_smooth_color(in, ini_i[i], ini_j[i], end_i[i], end_j[i]);
			}
		}
}

int main(int argc, char *argv[]) {
	int tipo_img = atoi(argv[2]);
	int NUM_NODES;
	MPI_Init(&argc, &argv);
	int rank;
    MPI_Status status;
    //descobre o rank do nó
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//descobre quantos nós a maquina suporta
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_NODES);

    Mat in, crop;
    int h_img_entrada, w_img_entrada;

    if(rank == 0)  {
    	//le a imagem  e salva suas dimensões para um futuro uso
    	if(tipo_img == 0) {
	    	in = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		} else if(tipo_img == 1) {
			in = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		}
	    h_img_entrada = in.size().height;
	    w_img_entrada = in.size().width;

	    //calcula o tamanho de cada pedaço
		int divisao_h = in.size().height/(NUM_NODES - 1);
		int divisao_w = in.size().width;
		int resto_divisao = in.size().height%(NUM_NODES - 1);
		//variavel auxiliar pra receber os dimensões de cada pedaço e manda pra cada nó
		int dimensoes[3];
		for(int i = 1; i < NUM_NODES; i++) {

			//recorto um pedaço da imagem pra mandar para o nó i
			//if else é pra pegar o restinho da imagem que sobre quando a divisão não é inteira
			if(i != NUM_NODES - 1) {
				crop =in(Rect(0,divisao_h*(i - 1),divisao_w,divisao_h));
			} else {
				crop =in(Rect(0,divisao_h*(i - 1),divisao_w,divisao_h + resto_divisao));
			}
			//pego as dimensões do pedaço da imagem e salvo num tipo struct 'image' definido la em cima
			dimensoes[0] = crop.size().width;
			dimensoes[1] = crop.size().height;
			dimensoes[2] = tipo_img;
			// envio as dimensões para o nó
   			MPI_Send(dimensoes, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
			// envio o pedaço da imagem'
			  if(tipo_img == 0){
				MPI_Send(crop.data, dimensoes[0]*dimensoes[1], MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
   			} else if(tipo_img == 1) {
   				MPI_Send(crop.data, dimensoes[0]*dimensoes[1]*3, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
   			}
		}
		//apago a imagem original para economizar memória
		in.release();

		//recebe os pedaços dos nós
		for(int i = 1; i < NUM_NODES; i++) {
			MPI_Recv(dimensoes, 3, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
			//não precisa ser global
			if(dimensoes[2] == 0) {
				Mat pedaco(dimensoes[1], dimensoes[0] ,CV_8U);
				MPI_Recv(pedaco.data, dimensoes[0]*dimensoes[1], MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, &status);
				if(i == 1) {
					vconcat(pedaco, out);
				} else {
					vconcat(out, pedaco, out);
				}
			} else if(dimensoes[2] == 1) {
				Mat pedaco(dimensoes[1], dimensoes[0] ,CV_8UC3);
				MPI_Recv(pedaco.data, dimensoes[0]*dimensoes[1]*3, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, &status);
				if(i == 1) {
					vconcat(pedaco, out);
				} else {
					vconcat(out, pedaco, out);
				}
			}
		}
		imwrite(argv[3], out);
	} else {
		//parte do código que roda dentro de cada nó
		int dimensoes[3];
		///recebe as dimensões do pedaço que sera manipulado
		MPI_Recv(dimensoes, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

		//recebe os dados do pedaco
		if(dimensoes[2] == 0) {
			//cria um mat do tamanho do pedaco;	
			Mat pedaco(dimensoes[1], dimensoes[0], CV_8U);	
			MPI_Recv(pedaco.data, dimensoes[0]*dimensoes[1], MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &status);
			//aplica a máscara no pedaco
			smooth(pedaco, dimensoes[2]);
			//libera a memoria
			pedaco.release();
		} else if(dimensoes[2] == 1) {
			//cria um mat do tamanho do pedaco;	
			Mat pedaco(dimensoes[1], dimensoes[0], CV_8UC3);	
			MPI_Recv(pedaco.data, dimensoes[0]*dimensoes[1]*3, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &status);
					//aplica a máscara no pedaco
			smooth(pedaco, dimensoes[2]);
			//libera a memoria
			pedaco.release();
		}
		dimensoes[0] = out.size().width;
		dimensoes[1] = out.size().height;
		MPI_Send(dimensoes, 3, MPI_INT, 0, 2, MPI_COMM_WORLD);
		if (dimensoes[2] == 0) {
			MPI_Send(out.data, dimensoes[0]*dimensoes[1], MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
		} else if (dimensoes[2] == 1) {
			MPI_Send(out.data, dimensoes[0]*dimensoes[1]*3, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
		}
	}
	MPI_Finalize();
	return 0;
}
