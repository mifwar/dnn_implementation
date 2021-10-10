#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/trace.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>      // std::setprecision
#include <cstdlib>

#include <vector>
#include <atomic>
#include <thread>

using namespace std;
using namespace cv;
using namespace cv::dnn;

atomic<bool> isFrameOkay;
int inc = 0;

// string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
// 	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
// 	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
// 	"sofa", "train", "tvmonitor"};

VideoCapture cap;
// VideoWriter vw;
Mat img;
bool activateThreadFrame = false;
bool keepThreadFrame = false;

string CLASSES[] = {"background", "MIFTAH"};
// string CLASSES[] = {"background", "goalpost", "x_line", "t_line", "l_line", "post"};

void thread_frame()
{
	while(activateThreadFrame)
	{
		if(!keepThreadFrame)
			continue;

		bool isFrameOkay_temp = cap.read(img);
		// bool isFrameOkay_temp = cap.read(imgtemp);
		// resize(imgtemp, img, Size(640, 360));
		isFrameOkay = isFrameOkay_temp;
	}
}

void frame_nodeflux_style(Mat frame, Rect rectku, int lineSize)
{
	line(frame, Point(rectku.x, rectku.y), Point(rectku.x + rectku.width/3, rectku.y), Scalar(0, 255, 0), lineSize);
	line(frame, Point(rectku.x + rectku.width/3*2, rectku.y), Point(rectku.x + rectku.width, rectku.y), Scalar(0, 255, 0), lineSize);

	line(frame, Point(rectku.x, rectku.y+rectku.height), Point(rectku.x + rectku.width/3, rectku.y+rectku.height), Scalar(0, 255, 0), lineSize);
	line(frame, Point(rectku.x + rectku.width/3*2, rectku.y+rectku.height), Point(rectku.x + rectku.width, rectku.y+rectku.height), Scalar(0, 255, 0), lineSize);

	line(frame, Point(rectku.x + rectku.width, rectku.y), Point(rectku.x + rectku.width, rectku.y+rectku.height/3), Scalar(0, 255, 0), lineSize);
	line(frame, Point(rectku.x + rectku.width, rectku.y+rectku.height/3*2), Point(rectku.x + rectku.width, rectku.y+rectku.height), Scalar(0, 255, 0), lineSize);

	line(frame, Point(rectku.x, rectku.y), Point(rectku.x, rectku.y+rectku.height/3), Scalar(0, 255, 0), lineSize);
	line(frame, Point(rectku.x, rectku.y+rectku.height/3*2), Point(rectku.x, rectku.y+rectku.height), Scalar(0, 255, 0), lineSize);
}

int main(int argc, char **argv)
{
	char typed = waitKey(1);

    CV_TRACE_FUNCTION();

	String tf_model_ori = "face/frozen_inference_graph_face.pb";
    String tf_config_ori = "face/output.pbtxt";

	cap.open(2);
	cap.set(cv::CAP_PROP_FPS, 30);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    
	Net net = dnn::readNetFromTensorflow(tf_model_ori, tf_config_ori);

    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << tf_model_ori << std::endl;
        std::cerr << "caffemodel: " << tf_config_ori << std::endl;
        exit(-1);
    }

	string stringku;
	char buffer[512];

	std::thread t_cam;
	t_cam = std::thread(thread_frame);
	activateThreadFrame = true;
	keepThreadFrame = true;
	Mat imgClone;
	while(true)
    {		

		if(!isFrameOkay.load())
			continue;

		Mat cloneImg = img.clone();
	    
	    Mat img2;
	    resize(cloneImg, img2, Size(300,300));
		Mat inputBlob = blobFromImage(img2, 1.0, Size(300,300), Scalar(127.5, 127.5, 127.5), true, false);

		net.setInput(inputBlob, "data");
	    Mat detection = net.forward("detection_out");
	    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	    ostringstream ss;
	    float confidenceThreshold = 0.4;

	    for (int i = 0; i < detectionMat.rows; i++)
	    {
	        float confidence = detectionMat.at<float>(i, 2);

	        if (confidence > confidenceThreshold)
	        {
				Mat blankFrame(cloneImg.size(), cloneImg.type(), cv::Scalar(0, 0, 0));
	            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
	            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * cloneImg.cols);
	            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * cloneImg.rows);
	            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * cloneImg.cols);
	            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * cloneImg.rows);

	            Rect object((int)xLeftBottom, (int)yLeftBottom,
	                        (int)(xRightTop - xLeftBottom),
	                        (int)(yRightTop - yLeftBottom));

				rectangle(blankFrame, object, Scalar(0, 255, 0), 2);

				if(object.width > object.height)
					continue;
	            // ss.str("");
				char buffer[8];
				sprintf(buffer, "%1.2f", confidence);
	            // ss << confidence;
	            String conf(buffer);
	            String label = CLASSES[idx];
	            String prob  = conf;
	            int baseLine = 0;
	            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
	            Size probSize = getTextSize(prob, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
				Rect nameRect(object.x + object.width, object.y, labelSize.width + 2, labelSize.height*1.5);
				Rect probRect(object.x + object.width, object.y + nameRect.height, probSize.width + 2, probSize.height*1.5);

				addWeighted(cloneImg, 1.0, blankFrame, 0.5, 0.0, cloneImg);

				Mat modelName(cloneImg.size(), cloneImg.type(), cv::Scalar(255, 255, 255));
				// Mat modelName = cloneImg.clone();
				rectangle(modelName, nameRect, Scalar(0, 0, 0), -1);
				rectangle(modelName, probRect, Scalar(0, 0, 0), -1);
				addWeighted(cloneImg, 0.4, modelName, 0.6, 0.0, imgClone);

				for(int i = 0; i < modelName.rows; i++)
				{
					for(int j = 0; j < modelName.cols; j++)
					{
						if(modelName.at<Vec3b>(i, j)[0] == 0 &&
						modelName.at<Vec3b>(i, j)[1] == 0 &&
						modelName.at<Vec3b>(i, j)[2] == 0)
							cloneImg.at<Vec3b>(i, j) = imgClone.at<Vec3b>(i, j);
					}
				}

				frame_nodeflux_style(cloneImg, object, 2);
	            putText(cloneImg, label, Point(nameRect.x + 2, nameRect.y + nameRect.height*0.75),
					FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255));
	                    // FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
				putText(cloneImg, prob, Point(probRect.x + 2, probRect.y + probRect.height*0.75),
	                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255));
	                    // FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
	        }
	    }
	    imshow("detections", cloneImg);

		// vw.write(img);

		isFrameOkay = false;
		if(waitKey(1) == 27)
		{
			activateThreadFrame = false;
			keepThreadFrame = false;
			t_cam.join();
			break;
		}
    }

	if(cap.isOpened())
		cap.release();

    return 0;
}