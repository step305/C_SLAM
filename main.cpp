#include <stdio.h>
#include <iostream>
#include <deque>
#include <stdlib.h>
#include <numeric>
#include <chrono>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

cv::Mat camMtx = (cv::Mat1d(3, 3) << 799.28791879, 0, 323.47630502, 0, 802.50512369, 246.49945916, 0, 0, 1);
cv::Mat distCoefs = (cv::Mat1d(1, 5) << -6.83033420e-02,  1.57869307e+00,  7.40633413e-03, -3.45875745e-03, -7.09892349e+0 );

int const npoints = 1000;
int const FPS = 30;
long long unsigned texp_ms = 20 * 1000;
long long unsigned tend;
const float  threshold_close = 2.0f;
const int  max_keypoints = 200;

bool response_comparator(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
{
    return p1.response > p2.response;
}

typedef struct {
    cv::Mat frame;
    cv::Mat descriptors;
    std::vector<cv::Point2f>  points;
    long long unsigned ts;
} CAMMessageStruct;

int main() {
    long long unsigned ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch( )
    ).count( );

    cv::Mat empty_frame;
    //Setup the Camera
    //cv::VideoCapture cap( 0 );
    cv::VideoCapture cap("v4l2src device=/dev/video0 do-timestamp=true ! video/x-raw, width=640, height=480, "
                         "framerate=30/1, format=NV12 ! videoconvert ! video/x-raw, format=BGR ! appsink");
    if( !cap.isOpened( ) )
    {
        std::cout << "Can not open the camera" << std::endl;
        abort();
    }

    //cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    //cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    //cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    //cap.set(cv::CAP_PROP_FPS, FPS);
    //cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);

    cv::VideoWriter video("video_slam.avi", cv::VideoWriter::fourcc('M','J','P','G'), FPS, cv::Size(640, 480), true);


    cv::Ptr<cv::ORB> orb_detector = cv::ORB::create( npoints );

    ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch( )
    ).count( );
    tend = ts + texp_ms;

    long long unsigned t0, te;

    t0 = ts;
    int fps_cnt = 0;

    while (ts < tend) {
        ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch( )
        ).count( );

        cv::Mat frame(640, 480, cv::DataType<float>::type);
        cap >> frame;

        if (frame.empty()){
            continue;
        }
        if (fps_cnt > 10) {
            te = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch( )
            ).count( );
            std::cout << "FPS = " << ((float)fps_cnt)/(float)(te-t0)*1000.0 << std::endl;
            fps_cnt = 0;
            t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch( )
            ).count( );
        } else {
            fps_cnt++;
        }

        cv::flip(frame,frame,0);
        cv::flip(frame,frame,1);

        video.write(frame);

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints;
        orb_detector->detect( frame, keypoints);

        if ( keypoints.size( ) > 0 ) {
            std::vector<cv::KeyPoint> keypoints_srt;
            std::sort(keypoints.begin(), keypoints.end(),
                      [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });

            for ( auto & keypoint : keypoints ){
                bool too_close = false;
                bool swapped = false;
                for ( auto & keypoint_srt : keypoints_srt )
                    if (threshold_close > cv::norm(keypoint.pt-keypoint_srt.pt)) {
                        if (keypoint.response > keypoint_srt.response) {
                            std::swap(keypoint, keypoint_srt);
                            swapped = true;
                        } else {
                            too_close = true;
                        }
                    }
                if (!too_close && !swapped)
                    keypoints_srt.push_back(keypoint);
            }
            cv::KeyPointsFilter::retainBest(keypoints_srt, max_keypoints);

            if (keypoints_srt.size() > 0) {
                cv::Mat descriptors;

                orb_detector->compute(frame, keypoints_srt, descriptors);

                std::vector<cv::Point2f> dist_points;
                std::vector<cv::Point2f> undist_points;
                cv::KeyPoint::convert(keypoints_srt, dist_points);
                cv::undistortPoints(dist_points, undist_points, camMtx, distCoefs);
            }

        }


    }


    return 0;
}
