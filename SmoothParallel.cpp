#include <iostream>
#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <omp.h>
#include <mpi.h>
#include <cstdio>
#include <cmath>

using namespace std;
using namespace cv;

#define SQUARE_AREA 25
//o processador é um quad core.
#define NUMTHREADS 4

Mat src, dst[3], dst2[3], finImg, aux, out;

void mediaBlurRGB(int inicio, int fim, int indice);
void mediaBlur(int inicio, int fim);
int display_finImg(int delay, String&, Mat&);

int main( int argc, char** argv) {

    int rank, top = 2, bottom = 2, left = 2, right = 2, borderType = BORDER_CONSTANT, tamLin, tamCol, bloco, resto, numNos;
    String window_name = "Smoothing ";

    MPI_Init(&argc, &argv);
    MPI_Status estado;
    
    //rank do nó.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //quantidade de nós que a máquina pode processar.
    MPI_Comm_size(MPI_COMM_WORLD, &numNos);
    
    //variavel auxiliar que conterá a dimensão de cada bloco da imagem e enviará para cada nó.
    int tamanhoBloco[3];

    //condição para entrar no nó master.
    if(rank == 0){

	    //Valores que serão inseridos na borda da imagem.
	    Scalar value( 0, 0, 0);

	    //Carrega a imagem ser usada.
	    int imgType = atoi(argv[2]);
	    src = imread( argv[1], imgType );
	     
	    //Cria a borda na imagem.
	    copyMakeBorder(src,src, top, bottom,left, right, borderType, value);
	  
	    //clona a imagem de entrada.
	    finImg = src.clone();

	    //divide a img em partes iguais para paralelizar no openmpi
	    int divisaoLinha = tamLin/(numNos - 1);
            int divisaoColuna = tamCol;
	    int restoDivisao = tamLin%(numNos - 1);

	    int i;
	    //paralelização dos blocos da imagem, enviando cada bloco para um nó diferente.
	    for(i=0;i< numNos;++i){

		//Neste "if else", recortamos um pedaço da img original que será salvo em 'aux' para depois enviar para os nós.
                if(i!=numNos -1){
			aux = src(Rect(0,divisaoLinha*(i-1), divisaoColuna, divisaoLinha));		
		}else{
			aux = src(Rect(0,divisaoLinha*(i-1), divisaoColuna, divisaoLinha + restoDivisao));		
		}

		tamanhoBloco[0] = aux.cols;
		tamanhoBloco[1] = aux.rows;
		tamanhoBloco[2] = imgType;

		//envia as o tamanho do bloco da imagem para o nó.
		MPI_Send(tamanhoBloco, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
		
		//envia o pedaço da imagem para o nó, dependendo do tipo da imagem.
		if(tamanhoBloco[2] == 0){
		   MPI_Send(aux.data, tamanhoBloco[0]*tamanhoBloco[1], MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
		}else if(tamanhoBloco[2] == 1){
		   MPI_Send(aux.data, tamanhoBloco[0]*tamanhoBloco[1]*3, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
		}

	    }
	
	    //Recebe cada pedaço de cada nó.
	    for(i=1; i<numNos; ++i){
		MPI_Recv(tamanhoBloco, 3, MPI_INT, i, 2, MPI_COMM_WORLD, &estado);
		
		//Verifica se a imagem é Grayscale ou RGB.
		if(tamanhoBloco[2] == 0){

		   Mat pedaco(tamanhoBloco[1], tamanhoBloco[0] ,CV_8U);
		   MPI_Recv(pedaco.data, tamanhoBloco[0]*tamanhoBloco[1], MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, &estado);
		   if(i == 1) {
			vconcat(pedaco, out);
		   } else {
			vconcat(out, pedaco, out);
		   }

		}else if(tamanhoBloco[2] == 1){
		   Mat pedaco(tamanhoBloco[1], tamanhoBloco[0] ,CV_8U);
		   MPI_Recv(pedaco.data, tamanhoBloco[0]*tamanhoBloco[1]*3, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, &estado);
		   if(i == 1) {
			vconcat(pedaco, out);
		   } else {
			vconcat(out, pedaco, out);
		   }
		}
	    }
	    
	    //salva a imagem no diretório.
 	    imwrite("novaImg.jpg", out);
    }else{	
	    //dentro de cada nó ele executará esse else.
	    MPI_Recv(tamanhoBloco, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &estado);
	
	    if(tamanhoBloco[2] == 0) {
		
		//cria um mat do tamanho do pedaco;	
		Mat pedaco(tamanhoBloco[1], tamanhoBloco[0], CV_8U);	
		MPI_Recv(pedaco.data, tamanhoBloco[0]*tamanhoBloco[1], MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &estado);
		
		tamLin = pedaco.rows;
	    	tamCol = pedaco.cols;

       	        //divide a img em partes iguais para paralelizar no openmp
		bloco = tamLin/NUMTHREADS;
		resto = tamLin%NUMTHREADS;

		//percorrer o loop para cada bloco da imagem definida.
		#pragma omp parallel for
		for(int i=0; i < NUMTHREADS; ++i){
		   if(i == NUMTHREADS-1)
		      mediaBlurGrayscale(bloco*i, (bloco*(i+1)) - 1);
		   else
		      mediaBlurGrayscale(bloco*i, (bloco*(i+1)) - 1 + resto);
		}

                //Retira a borda criada anteriormente da imagem.
		finImg = finImg.colRange(3, (finImg.cols-3));
		finImg = finImg.rowRange(3, (finImg.rows-3));
		
		//libera espaço na memória
		pedaco.release();

	    } else {

		    //Separa a imagem original nos três canais disponíveis(RGB).
		    split(src,dst);
		    split(src, dst2);
		    
		    Mat pedaco(tamanhoBloco[1], tamanhoBloco[0], CV_8UC3);	
		    MPI_Recv(pedaco.data, tamanhoBloco[0]*tamanhoBloco[1]*3, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &estado);
			
		    tamLin = pedaco.rows;
	    	    tamCol = pedaco.cols;

       	            //divide a img em partes iguais para paralelizar no openmp.
		    bloco = tamLin/NUMTHREADS;
		    resto = tamLin%NUMTHREADS;

		    //Aplica MediaBlur para cada bloco da img.
		    for(int j=0;j<3;++j){

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


		    //Coloca os canais RGB dentro da matriz rgb.
		    vector<Mat> rgb;
		    rgb.push_back(dst[0]); //blue
		    rgb.push_back(dst[1]); //green
		    rgb.push_back(dst[2]); //red


		//Junta os três canais da imagem na imagem final.
		merge(rgb, finImg);
		pedaco.release();
	    }
		
	    tamanhoBloco[0] = finImg.size().width;
	    tamanhoBloco[1] = finImg.size().height;
	    
	    MPI_Send(tamanhoBloco, 3, MPI_INT, 0, 2, MPI_COMM_WORLD);
	    //Envia a imagem final para o master.
	    if(tamanhoBloco[2]==0){
	    	MPI_Send(out.data, tamanhoBloco[0]*tamanhoBloco[1], MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
            }else if(tamanhoBloco[2]==1){
		MPI_Send(out.data, tamanhoBloco[0]*tamanhoBloco[1]*3, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
	    }
    }
    //FInaliza o o OpenMPI.
    MPI_Finalize();
    return 0;
}

//Método de aplicação do mediaBlur para as imagens em RGB.
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


//Método de aplicação do mediaBlur para as imagens em Grayscale.
void mediaBlurGrayscale(int inicio, int fim){
    
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
                    sum += src.at<uchar>(l,c);

                }
            }
	media = sum/SQUARE_AREA;
        finImg.at<uchar>(i,j) = media;
        }
    }
}
