#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;

#define CLIP(x) x > 255 ? 255 : x < 0 ?  0 : x
void Histogram_MakeHistogram(Mat&HistImgbuf, float HistArray[256], int width, int height, bool bRGB = true)
{
	//8-bit depth
	int Image_Histogram[256] = { 0, };
	// 히스토그램 계산
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Image_Histogram[HistImgbuf.at<uchar>(i, j)]++;
		}
	}

	//int sub = Image_Histogram[0];
	//if (!bRGB)Image_Histogram[0] = 0;

	//히스토그램 정규화
	float Image_Area = (float)(height*width);
	for (int i = 0; i < 256; i++) {
		HistArray[i] = Image_Histogram[i] / (Image_Area);
	}
}


void calc_Histo(const Mat& image, Mat&hist, int bins, int range_max = 256) {

	int histSize[] = { bins };
	float range[] = { 0, (float)range_max };
	int channels[] = { 0 };
	const float* ranges[] = { range };

	calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
}

void draw_histo(Mat hist, Mat &hist_img, Size size = Size(256, 200)) {
	hist_img = Mat(size, CV_8U, Scalar(255));
	float bin = (float)hist_img.cols / hist.rows;
	normalize(hist, hist, 0, hist_img.rows, NORM_MINMAX);

	for (int i = 0; i < hist.rows; i++) {
		float start_x = i * bin;
		float end_x = (i + 1)*bin;
		Point2f pt1(start_x, 0);
		Point2f pt2(end_x, hist.at <float>(i));
		if (pt2.y > 0)
			rectangle(hist_img, pt1, pt2, Scalar(0), -1);
	}
	flip(hist_img, hist_img, 0);
}

void create_his(Mat img, Mat&hist, Mat &hist_img) {
	int histsize = 256, range = 256;
	calc_Histo(img, hist, histsize, range);
	draw_histo(hist, hist_img);
}


Mat Histogram_Match(Mat &dst, Mat &srcimg, int width, int height, float *Ref_Histogram) {
	float lookup[256] = { 0, };
	float Image_Histogram[256] = { 0, };
	
	Histogram_MakeHistogram(srcimg, Image_Histogram, width, height);
	// 히스토그램 누적 합 계산
	double CDF[256] = { 0.0, };
	CDF[0] = Image_Histogram[0];
	for (int i = 1; i < 256; i++) {
		CDF[i] = CDF[i - 1] + Image_Histogram[i];
	}
	

	// 히스토그램 누적 합 계산
	double CDF2[256] = { 0.0, };
	CDF2[0] = Ref_Histogram[0];

	for (int i = 1; i < 256; i++) {
		CDF2[i] = CDF2[i - 1] + Ref_Histogram[i];
	}
	int k;
	for (int i = 1; i < 256; i++) {
		k = 255;
		while (round(255 * CDF2[k])  > round(255 * CDF[i])) {
			k--;
		}
		if (k<0)
			lookup[i] = 0;
		else
			lookup[i] = k;
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = CLIP(lookup[srcimg.at<uchar>(i, j)]);
		}
	}

	return dst;
}


float* MakeRefHisto() {
	static float Ref_Histogram[256] = { 0, };
	float pi = 3.141592;
	float m1 = 0.1, m2 = 0.25, sig1 = 0.04, sig2 = 0.04, A1 = 1, A2 = 0.07, K = 0.1;
	float sum = 0, max = 0;
	for (int i = 1; i < 256; i++) {
		Ref_Histogram[i] = (A1 / (sqrt(2 * pi)*sig1)) * exp(-(float(i / 255.0) - m1)*(float(i / 255.0) - m1) / (2 * sig1*sig1))
			+ (A2 / (sqrt(2 * pi)*sig2)) * exp(-(float(i / 255.0) - m2)*(float(i / 255.0) - m2) / (2 * sig2*sig2)) + K;
		if (Ref_Histogram[i] > max) max = Ref_Histogram[i];
		sum += Ref_Histogram[i];
	}
	Ref_Histogram[0] = max;
	sum += max;
	for (int i = 0; i < 256; i++)
		Ref_Histogram[i] /= sum;

	return Ref_Histogram;
}
int main(void)
{
	VideoCapture Video;
	VideoWriter outputVideo;
	outputVideo.open("Output.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), 30.f, Size(640, 512), true);
	if (!outputVideo.isOpened()) return -1;

	String filename;
	filename = "IO0000";
	char*temp = new char[6];
	filename = temp;

	for (int i = 1600; i < 2500; i++) {

		Mat hist_img, hist;
		String newfilename;
		if (i < 10) {
			_itoa_s(i, temp, 5, 10);
			newfilename = (String)"I0000" + (String)(temp);
		}
		else if (i >= 10 && i < 100) {
			_itoa_s(i, temp, 5, 10);
			newfilename = (String)"I000" + (String)(temp);
		}
		else if (i >= 100 && i < 1000) {
			_itoa_s(i, temp, 5, 10);
			newfilename = (String)"I00" + (String)(temp);
		}
		else if (i >= 1000 && i < 10000) {
			_itoa_s(i, temp, 5, 10);
			newfilename = (String)"I0" + (String)(temp);
		}
		cout << newfilename << endl;

		String Colorpath, Thermalpath;
		Colorpath = "./visible/" + newfilename + ".jpg";
		Thermalpath = "./lwir/" + newfilename + ".jpg";

		Mat Color, thumer;
		Color = imread(Colorpath, CV_LOAD_IMAGE_COLOR);
		thumer = imread(Thermalpath, CV_LOAD_IMAGE_COLOR);
		
		Mat yuvImgC;
		Mat yuvImgT;

		cvtColor(Color, yuvImgC, CV_BGR2YUV);
		vector<Mat> yuvC_images(3);
		split(yuvImgC, yuvC_images);
		cvtColor(thumer, yuvImgT, CV_BGR2YUV);
		vector<Mat> yuvT_images(3);
		split(yuvImgT, yuvT_images);
		vector<Mat> yuvCT_images(3);
		Mat dst = Mat::zeros(512, 640, CV_8U);
		Mat dst1 = Mat::zeros(512, 640, CV_8U);

		float* Ref_Histogram = MakeRefHisto();
		Histogram_Match(dst, yuvT_images[0], 640, 512, Ref_Histogram);
		yuvC_images[0] = dst;
		
		// 히스토그램 뷰어
		/*
		calc_Histo(dst, hist, 256);
		draw_histo(hist, hist_img);
		imshow("frame", hist_img);
		*/

		Mat dest2;
		Mat Merge;
		merge(yuvC_images, Merge);
		cvtColor(Merge, dest2, CV_YUV2BGR);
		imshow("frame", dest2);
		outputVideo.write(dest2);
		waitKey(30);

	}

	waitKey(0);
	outputVideo.release();
	return 0;
}

