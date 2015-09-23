// testOpenCV.cpp : Defines the entry point for the console application.
//

//#include "stdio.h"
#include "../../libs/image_utils.h"
//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;


void processVideo(char* videoFilename);

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: trafficMonitoring video.avi" << endl;
		return -1;
	}
	//create GUI windows
	namedWindow("Frame");

	//Process the Video
	processVideo(argv[1]);

	waitKey(0); // Wait for a keystroke in the window
	//destroy GUI windows
	destroyAllWindows();
	return EXIT_SUCCESS;

}
void mergeMask(Mat& mask)
{
	int type = MORPH_ELLIPSE;
	int size = 2;
	Mat element = getStructuringElement(type,
		Size(2 * size + 1, 2 * size + 1),
		Point(size, size));
	morphologyEx(mask, mask, MORPH_OPEN, element, Point(-1, -1), 2);
	morphologyEx(mask, mask, MORPH_CLOSE, element, Point(-1, -1), 4);
}
void addInfo(Mat& frame, string string)
{
	rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
		cv::Scalar(255, 255, 255), -1);
	putText(frame, string.c_str(), cv::Point(15, 15),
		FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
}
void addBoundingBox(Mat& frame, Mat& mask)
{
	vector< vector< cv::Point> > contours;
	Mat threshout = mask.clone();
	//cvtColor(threshout, threshout, COLOR_BGR2GRAY);
	//threshold(threshout, threshout, 1, 255, CV_THRESH_BINARY);
	findContours(threshout, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); //Use in Release mode only!!!

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	/// Draw polygonal contour + bonding rects + circles
	//cvtColor(frame, frame, CV_GRAY2RGB);
	RNG rng(12345);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
	}

}

void processVideo(char* videoFilename)
{
	//Getting the VideoCapture object
	VideoCapture capture(videoFilename);
	if (!capture.isOpened())
	{
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}

	//create Background Subtractor objects
	int history = 500;
	double thresh = 16.0;
	bool detectShadows = false;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(history, thresh, detectShadows); //MOG2 approach

	//read input data and process
	int keyboard = 0;
	Mat orig, frame, fgMaskMOG2;
	while ((char)keyboard != 'q' && (char)keyboard != 27)
	{
		if (keyboard == 'p')
		{
			waitKey(0);
		}
		//Reading the frame
		if (!capture.read(orig))
		{
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}
		//Some enhancements
		cvtColor(orig, frame, COLOR_BGR2GRAY);
		GaussianBlur(frame, frame, Size(5, 5), 1.5, 1.5);


		//update the background model and gets forground mask
		pMOG2->apply(frame, fgMaskMOG2);

		//get the frame number and write it on the current frame
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES) << "/" << capture.get(CV_CAP_PROP_FRAME_COUNT);
		addInfo(orig, ss.str());

		//Enhance the mask
		mergeMask(fgMaskMOG2);

		/// Find contours
		//Mat cimg = frame.clone();
		addBoundingBox(orig, fgMaskMOG2);

		//show the current frame and the fg masks
		imshow("Frame", orig);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = waitKey(30);
	}

	//delete capture object
	capture.release();
}