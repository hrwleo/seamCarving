#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include "cstringt.h"
#include "io.h"
#include <string>
#include <vector>
#include <fstream>
using namespace std;
#define NUM 100

vector<string> filenamesGlobal;

vector<string> getFilesName(string path){
	//�ļ����  
	long   hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	string p;
	vector<string>filenames;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFilesName(p.assign(path).append("\\").append(fileinfo.name));
			}
			else
			{
				filenames.push_back(fileinfo.name);  // ֱ�ӻ�ȡ���ּ���
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name));  
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return filenames;
}

void calculateEnergy(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& traceMat)
{
	srcMat.copyTo(dstMat);   //���á�=������ֹ��������ָ��Ķ���ͬһ����������ֻ��Ҫ���������ֵ

	for (int i = 1;i < srcMat.rows;i++)  //�ӵ�2�п�ʼ����
	{
		//��һ��
		if (dstMat.at<float>(i-1,0) <= dstMat.at<float>(i-1,1))
		{
			dstMat.at<float>(i,0) = srcMat.at<float>(i,0) + dstMat.at<float>(i-1,0);
			traceMat.at<float>(i,0) = 1; //traceMat��¼��ǰλ�õ���һ��Ӧȡ�Ǹ�λ�ã�����Ϊ0������1������Ϊ2
		}
		else
		{
			dstMat.at<float>(i,0) = srcMat.at<float>(i,0) + dstMat.at<float>(i-1,1);
			traceMat.at<float>(i,0) = 2;
		}
		
		//�м���
		for (int j = 1;j < srcMat.cols-1;j++)
		{
			float k[3];
			k[0] = dstMat.at<float>(i-1,j-1);
			k[1] = dstMat.at<float>(i-1,j);
			k[2] = dstMat.at<float>(i-1,j+1);

			int index = 0;
			if (k[1] < k[0])
				index = 1;
			if (k[2] < k[index])
				index = 2; 
			dstMat.at<float>(i,j) = srcMat.at<float>(i,j) + dstMat.at<float>(i-1,j-1+index);
			traceMat.at<float>(i,j) = index;

		}

		//���һ��
		if (dstMat.at<float>(i-1,srcMat.cols-1) <= dstMat.at<float>(i-1,srcMat.cols-2))
		{
			dstMat.at<float>(i,srcMat.cols-1) = srcMat.at<float>(i,srcMat.cols-1) + dstMat.at<float>(i-1,srcMat.cols-1);
			traceMat.at<float>(i,srcMat.cols-1) = 1; 
		}
		else
		{
			dstMat.at<float>(i,srcMat.cols-1) = srcMat.at<float>(i,srcMat.cols-1) + dstMat.at<float>(i-1,srcMat.cols-2);
			traceMat.at<float>(i,srcMat.cols-1) = 0;
		}

	}
}

void calculateEnergy_row(cv::Mat& srcMat, cv::Mat& dstMat, cv::Mat& traceMat)
{
	srcMat.copyTo(dstMat);   //���á�=������ֹ��������ָ��Ķ���ͬһ����������ֻ��Ҫ���������ֵ

	for (int i = 1; i < srcMat.cols; i++)  //�ӵ�2�п�ʼ����
	{
		//��һ��
		if (dstMat.at<float>(0, i - 1) <= dstMat.at<float>(1, i - 1))
		{
			dstMat.at<float>(0, i) = srcMat.at<float>(0, i) + dstMat.at<float>(0, i - 1);
			traceMat.at<float>(0, i) = 1; //traceMat��¼��ǰλ�õ���һ��Ӧȡ�Ǹ�λ�ã�����Ϊ0������1������Ϊ2
		}
		else
		{
			dstMat.at<float>(0, i) = srcMat.at<float>(0, i) + dstMat.at<float>(1, i - 1);
			traceMat.at<float>(0, i) = 2;
		}

		//�м���
		for (int j = 1; j < srcMat.rows - 1; j++)
		{
			float k[3];
			k[0] = dstMat.at<float>(i - 1, j - 1);
			k[1] = dstMat.at<float>(i, j - 1);
			k[2] = dstMat.at<float>(i + 1, j - 1);

			int index = 0;
			if (k[1] < k[0])
				index = 1;
			if (k[2] < k[index])
				index = 2;
			dstMat.at<float>(i, j) = srcMat.at<float>(i, j) + dstMat.at<float>(i - 1, j - 1 + index);
			traceMat.at<float>(i, j) = index;

		}

		//���һ��
		if (dstMat.at<float>(srcMat.rows - 1, i - 1) <= dstMat.at<float>(srcMat.rows - 2, i - 1))
		{
			dstMat.at<float>(srcMat.rows - 1, i) = srcMat.at<float>(srcMat.rows - 1, i) + dstMat.at<float>(srcMat.rows - 1, i - 1);
			traceMat.at<float>(srcMat.rows - 1, i) = 1;
		}
		else
		{
			dstMat.at<float>(srcMat.rows - 1, i) = srcMat.at<float>(srcMat.rows - 1, i) + dstMat.at<float>(srcMat.rows - 2, i - 1);
			traceMat.at<float>(srcMat.rows - 1, i) = 0;
		}

	}
}

// �ҳ���С������
void getMinEnergyTrace(const cv::Mat& energyMat,const cv::Mat& traceMat,cv::Mat& minTrace)
{
	int row = energyMat.rows - 1;// ȡ����energyMat���һ�е����ݣ������б���rows-1

	int index = 0;	// ���������С�����켣�����������ͼ���е��б�

	// ���index�������������Сֵ��λ��
	for (int i = 1;i < energyMat.cols;i++)
	{
		if (energyMat.at<float>(row,i) < energyMat.at<float>(row,index))
		{
			index = i;
		} // end if
	} // end for i = ...
	
	// ���¸���traceMat���õ�minTrace��minTrace�Ƕ���һ�о���
	{
		minTrace.at<float>(row,0) = index;

		int tmpIndex = index;

		for (int i = row;i > 0;i--)
		{
			int temp = traceMat.at<float>(i,tmpIndex);// ��ǰλ��traceMat�����ֵ

			if (temp == 0) // ������
			{
				tmpIndex = tmpIndex - 1;
			}
			else if (temp == 2) // ������
			{
				tmpIndex = tmpIndex + 1;
			} // ���temp = 1�����������ߣ�tmpIndex����Ҫ���޸�

			minTrace.at<float>(i-1,0) = tmpIndex;
		}
	}
}

void getMinEnergyTrace_row(const cv::Mat& energyMat, const cv::Mat& traceMat, cv::Mat& minTrace)
{
	int col = energyMat.cols - 1;// ȡ����energyMat���һ�е����ݣ������б���rows-1

	int index = 0;	// ���������С�����켣�����������ͼ���е��б�

	// ���index�������������Сֵ��λ��
	for (int i = 1; i < energyMat.rows; i++)
	{
		if (energyMat.at<float>(i, col) < energyMat.at<float>(index, col))
		{
			index = i;
		} // end if
	} // end for i = ...

	// ���¸���traceMat���õ�minTrace��minTrace�Ƕ���һ�о���
	{
		minTrace.at<float>(0, col) = index;

		int tmpIndex = index;

		for (int i = col; i > 0; i--)
		{
			int temp = traceMat.at<float>(i, tmpIndex);// ��ǰλ��traceMat�����ֵ

			if (temp == 0) // ������
			{
				tmpIndex = tmpIndex - 1;
			}
			else if (temp == 2) // ������
			{
				tmpIndex = tmpIndex + 1;
			} // ���temp = 1�����������ߣ�tmpIndex����Ҫ���޸�

			minTrace.at<float>(0, i - 1) = tmpIndex;
		}
	}
}

// ɾ��һ��
void delOneCol(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& minTrace,cv::Mat& beDeletedLine)
{
	
	for (int i = 0;i < dstMat.rows;i++)
	{
		int k = minTrace.at<float>(i,0);
		
		for (int j = 0;j < k;j++)
		{
			dstMat.at<cv::Vec3b>(i,j)[0] = srcMat.at<cv::Vec3b>(i,j)[0];
			dstMat.at<cv::Vec3b>(i,j)[1] = srcMat.at<cv::Vec3b>(i,j)[1];
			dstMat.at<cv::Vec3b>(i,j)[2] = srcMat.at<cv::Vec3b>(i,j)[2];
		}
		for (int j = k;j < dstMat.cols-1;j++)
		{
			if (j == dstMat.cols-1)
			{
				int a = 1;
			}
			dstMat.at<cv::Vec3b>(i,j)[0] = srcMat.at<cv::Vec3b>(i,j+1)[0];
			dstMat.at<cv::Vec3b>(i,j)[1] = srcMat.at<cv::Vec3b>(i,j+1)[1];
			dstMat.at<cv::Vec3b>(i,j)[2] = srcMat.at<cv::Vec3b>(i,j+1)[2];

		}
		{
			beDeletedLine.at<cv::Vec3b>(i,0)[0] = srcMat.at<cv::Vec3b>(i,k)[0];
			beDeletedLine.at<cv::Vec3b>(i,0)[1] = srcMat.at<cv::Vec3b>(i,k)[1];
			beDeletedLine.at<cv::Vec3b>(i,0)[2] = srcMat.at<cv::Vec3b>(i,k)[2];
		}
	}
}


void delOneRow(cv::Mat& srcMat, cv::Mat& dstMat, cv::Mat& minTrace, cv::Mat& beDeletedLine)
{

	for (int i = 0; i < dstMat.cols; i++)
	{
		int k = minTrace.at<float>(i, 0);

		for (int j = 0; j < k; j++)
		{
			dstMat.at<cv::Vec3b>(i, j)[0] = srcMat.at<cv::Vec3b>(i, j)[0];
			dstMat.at<cv::Vec3b>(i, j)[1] = srcMat.at<cv::Vec3b>(i, j)[1];
			dstMat.at<cv::Vec3b>(i, j)[2] = srcMat.at<cv::Vec3b>(i, j)[2];
		}
		for (int j = k; j < dstMat.rows - 1; j++)
		{
			if (j == dstMat.rows - 1)
			{
				int a = 1;
			}
			dstMat.at<cv::Vec3b>(i, j)[0] = srcMat.at<cv::Vec3b>(i + 1, j)[0];
			dstMat.at<cv::Vec3b>(i, j)[1] = srcMat.at<cv::Vec3b>(i + 1, j)[1];
			dstMat.at<cv::Vec3b>(i, j)[2] = srcMat.at<cv::Vec3b>(i + 1, j)[2];

		}
		{
			beDeletedLine.at<cv::Vec3b>(i, 0)[0] = srcMat.at<cv::Vec3b>(i, k)[0];
			beDeletedLine.at<cv::Vec3b>(i, 0)[1] = srcMat.at<cv::Vec3b>(i, k)[1];
			beDeletedLine.at<cv::Vec3b>(i, 0)[2] = srcMat.at<cv::Vec3b>(i, k)[2];
		}
	}
}

int g_c = 0;
void run(cv::Mat& image,cv::Mat& outImage,cv::Mat& outMinTrace,cv::Mat& outDeletedLine)
{
	cv::Mat image_gray(image.rows, image.cols, CV_8U, cv::Scalar(0));
	cv::cvtColor(image, image_gray, CV_BGR2GRAY); //????????????????

	cv::Mat gradiant_H(image.rows, image.cols, CV_32F, cv::Scalar(0));//????????
	cv::Mat gradiant_V(image.rows, image.cols, CV_32F, cv::Scalar(0));//?????????

	cv::Mat kernel_H = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //???????????????????????????
	cv::Mat kernel_V = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //????????????????????????????

	cv::filter2D(image_gray, gradiant_H, gradiant_H.depth(), kernel_H);
	cv::filter2D(image_gray, gradiant_V, gradiant_V.depth(), kernel_V);

		cv::Mat gradMag_mat(image.rows, image.rows, CV_32F, cv::Scalar(0));
		cv::add(cv::abs(gradiant_H), cv::abs(gradiant_V), gradMag_mat);
		//??????????
		cv::Mat energyMat(image.rows, image.cols, CV_32F, cv::Scalar(0));//???????????
		cv::Mat traceMat(image.rows, image.cols, CV_32F, cv::Scalar(0));//??????��??????
		calculateEnergy(gradMag_mat, energyMat, traceMat);

		//?????��??????
		cv::Mat minTrace(image.rows, 1, CV_32F, cv::Scalar(0));//??????��???????��???��????????
		getMinEnergyTrace(energyMat, traceMat, minTrace);
		//?????��??????
		cv::Mat tmpImage(image.rows, image.cols, image.type());
		image.copyTo(tmpImage);
		for (int i = 0; i < image.rows; i++)
		{
			int k = minTrace.at<float>(i, 0);
			tmpImage.at<cv::Vec3b>(i, k)[0] = 0;
			tmpImage.at<cv::Vec3b>(i, k)[1] = 0;
			tmpImage.at<cv::Vec3b>(i, k)[2] = 255;
		}
		//cv::imshow("Image Show Window (A)", tmpImage);
		cv::Mat image2(image.rows, image.cols - 1, image.type());
		cv::Mat beDeletedLine(image.rows, 1, CV_8UC3);//??????????????��??
		delOneCol(image, image2, minTrace, beDeletedLine);
		//cv::imshow("Image Show Window", image2);

		image2.copyTo(outImage);
		minTrace.copyTo(outMinTrace);
		beDeletedLine.copyTo(outDeletedLine);
	
}

void recoverOneLine(cv::Mat& inImage,cv::Mat&inTrace,cv::Mat& inDeletedLine,cv::Mat& outImage)
{
	
	cv::Mat recorvedImage(inImage.rows,inImage.cols+1,CV_8UC3);
	for (int i = 0; i < inImage.rows; i++)
	{
		int k = inTrace.at<float>(i);
		for (int j = 0; j < k; j++)
		{
			recorvedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j)[0];
			recorvedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j)[1];
			recorvedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j)[2];
		}
		recorvedImage.at<cv::Vec3b>(i,k)[0] = inDeletedLine.at<cv::Vec3b>(i,0)[0];
		recorvedImage.at<cv::Vec3b>(i,k)[1] = inDeletedLine.at<cv::Vec3b>(i,0)[1];
		recorvedImage.at<cv::Vec3b>(i,k)[2] = inDeletedLine.at<cv::Vec3b>(i,0)[2];

		for (int j = k + 1;j < inImage.cols + 1; j++)
		{
			recorvedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j-1)[0];
			recorvedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j-1)[1];
			recorvedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j-1)[2];
		}
	}

	//��ʾ�ָ��Ĺ켣
	cv::Mat tmpImage(recorvedImage.rows,recorvedImage.cols,recorvedImage.type());
	recorvedImage.copyTo(tmpImage);
	for (int i = 0;i < tmpImage.rows;i++)
	{
		int k = inTrace.at<float>(i,0);
		tmpImage.at<cv::Vec3b>(i,k)[0] = 0;
		tmpImage.at<cv::Vec3b>(i,k)[1] = 255;
		tmpImage.at<cv::Vec3b>(i,k)[2] = 0;
	}
	//cv::imshow("Image Show Window (B)",tmpImage);

	recorvedImage.copyTo(outImage);
}

int main(int argc,char* argv)
{
	char filePath[150] = "E:\\newCAS-PEAL\\test_224";    //********************
	filenamesGlobal = getFilesName(filePath);
	for (int i = 0; i < 986913; i++)
	{
		char imgPath[150];
		strcpy(imgPath, filePath);
		strcat(imgPath, "\\");
		strcat(imgPath, filenamesGlobal[i].c_str());

		cv::Mat image = cv::imread(imgPath);
		//cv::namedWindow("Original Image");
		//cv::imshow("Original Image",image);

		cv::Mat tmpMat;
		image.copyTo(tmpMat);

		cv::Mat traces[NUM];
		cv::Mat deletedLines[NUM];

		cv::Mat outImage;

		cv::waitKey(20);

		for (int i = 0; i < NUM; i++)
		{
			run(tmpMat, outImage, traces[i], deletedLines[i]);
			tmpMat = outImage;
			//cv::waitKey(50);
		}
		IplImage imgTmp = outImage;
		IplImage *saveImg = cvCloneImage(&imgTmp);
		char imgSavePath[150] = "C:\\Users\\Stefan\\Desktop\\res\\";  //********************
		strcat(imgSavePath, filenamesGlobal[i].c_str());

		cvSaveImage(imgSavePath, saveImg);

		cv::Mat tmpMat2;
		outImage.copyTo(tmpMat2);

		/*for (int i = 0; i < NUM; i++)
		{

		recoverOneLine(tmpMat2,traces[NUM-i-1],deletedLines[NUM-i-1],outImage);
		tmpMat2 = outImage;
		cv::waitKey(50);
		}
		cv::waitKey(115000);*/
	}
	return 0;

}

















