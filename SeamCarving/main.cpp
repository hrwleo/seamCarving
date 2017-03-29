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
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	vector<string>filenames;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFilesName(p.assign(path).append("\\").append(fileinfo.name));
			}
			else
			{
				filenames.push_back(fileinfo.name);  // 直接获取名字即可
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name));  
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return filenames;
}

void calculateEnergy(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& traceMat)
{
	srcMat.copyTo(dstMat);   //不用“=”，防止两个矩阵指向的都是同一个矩阵，现在只需要传里面的数值

	for (int i = 1;i < srcMat.rows;i++)  //从第2行开始计算
	{
		//第一列
		if (dstMat.at<float>(i-1,0) <= dstMat.at<float>(i-1,1))
		{
			dstMat.at<float>(i,0) = srcMat.at<float>(i,0) + dstMat.at<float>(i-1,0);
			traceMat.at<float>(i,0) = 1; //traceMat记录当前位置的上一行应取那个位置，上左为0，上中1，上右为2
		}
		else
		{
			dstMat.at<float>(i,0) = srcMat.at<float>(i,0) + dstMat.at<float>(i-1,1);
			traceMat.at<float>(i,0) = 2;
		}
		
		//中间列
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

		//最后一列
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
	srcMat.copyTo(dstMat);   //不用“=”，防止两个矩阵指向的都是同一个矩阵，现在只需要传里面的数值

	for (int i = 1; i < srcMat.cols; i++)  //从第2行开始计算
	{
		//第一列
		if (dstMat.at<float>(0, i - 1) <= dstMat.at<float>(1, i - 1))
		{
			dstMat.at<float>(0, i) = srcMat.at<float>(0, i) + dstMat.at<float>(0, i - 1);
			traceMat.at<float>(0, i) = 1; //traceMat记录当前位置的上一行应取那个位置，上左为0，上中1，上右为2
		}
		else
		{
			dstMat.at<float>(0, i) = srcMat.at<float>(0, i) + dstMat.at<float>(1, i - 1);
			traceMat.at<float>(0, i) = 2;
		}

		//中间列
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

		//最后一列
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

// 找出最小能量线
void getMinEnergyTrace(const cv::Mat& energyMat,const cv::Mat& traceMat,cv::Mat& minTrace)
{
	int row = energyMat.rows - 1;// 取的是energyMat最后一行的数据，所以行标是rows-1

	int index = 0;	// 保存的是最小那条轨迹的最下面点在图像中的列标

	// 获得index，即最后那行最小值的位置
	for (int i = 1;i < energyMat.cols;i++)
	{
		if (energyMat.at<float>(row,i) < energyMat.at<float>(row,index))
		{
			index = i;
		} // end if
	} // end for i = ...
	
	// 以下根据traceMat，得到minTrace，minTrace是多行一列矩阵
	{
		minTrace.at<float>(row,0) = index;

		int tmpIndex = index;

		for (int i = row;i > 0;i--)
		{
			int temp = traceMat.at<float>(i,tmpIndex);// 当前位置traceMat所存的值

			if (temp == 0) // 往左走
			{
				tmpIndex = tmpIndex - 1;
			}
			else if (temp == 2) // 往右走
			{
				tmpIndex = tmpIndex + 1;
			} // 如果temp = 1，则往正上走，tmpIndex不需要做修改

			minTrace.at<float>(i-1,0) = tmpIndex;
		}
	}
}

void getMinEnergyTrace_row(const cv::Mat& energyMat, const cv::Mat& traceMat, cv::Mat& minTrace)
{
	int col = energyMat.cols - 1;// 取的是energyMat最后一行的数据，所以行标是rows-1

	int index = 0;	// 保存的是最小那条轨迹的最下面点在图像中的列标

	// 获得index，即最后那行最小值的位置
	for (int i = 1; i < energyMat.rows; i++)
	{
		if (energyMat.at<float>(i, col) < energyMat.at<float>(index, col))
		{
			index = i;
		} // end if
	} // end for i = ...

	// 以下根据traceMat，得到minTrace，minTrace是多行一列矩阵
	{
		minTrace.at<float>(0, col) = index;

		int tmpIndex = index;

		for (int i = col; i > 0; i--)
		{
			int temp = traceMat.at<float>(i, tmpIndex);// 当前位置traceMat所存的值

			if (temp == 0) // 往左走
			{
				tmpIndex = tmpIndex - 1;
			}
			else if (temp == 2) // 往右走
			{
				tmpIndex = tmpIndex + 1;
			} // 如果temp = 1，则往正上走，tmpIndex不需要做修改

			minTrace.at<float>(0, i - 1) = tmpIndex;
		}
	}
}

// 删掉一列
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
		cv::Mat traceMat(image.rows, image.cols, CV_32F, cv::Scalar(0));//??????С??????
		calculateEnergy(gradMag_mat, energyMat, traceMat);

		//?????С??????
		cv::Mat minTrace(image.rows, 1, CV_32F, cv::Scalar(0));//??????С???????е???С????????
		getMinEnergyTrace(energyMat, traceMat, minTrace);
		//?????С??????
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
		cv::Mat beDeletedLine(image.rows, 1, CV_8UC3);//??????????????е??
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

	//显示恢复的轨迹
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

















