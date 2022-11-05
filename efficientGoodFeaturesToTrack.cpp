#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace std::chrono;
vector<double> convolution(const cv::Mat& image, const vector<cv::KeyPoint>& pts, const cv::Mat& kernal)
{
    double pixSum = 0;
    vector<double> grad;
    for(int i = 0; i < pts.size(); i++)
    {
        int row = floor(pts[i].pt.y);
        int col = floor(pts[i].pt.x);
        //the pixel locates in the middle of the kernal 
        for (int k = 0; k < kernal.rows; k++) 
            for (int l = 0; l < kernal.cols; l++)
                pixSum += kernal.at<double>(k, l)*double(image.at<uchar>(k+row-1, l+col-1));
        grad.push_back(abs(pixSum));
        pixSum = 0;
    }

    return grad;
}

typedef pair<int, double> PAIR;
bool cmp_by_value(const PAIR& lhs, const PAIR& rhs)
{
    return lhs.second > rhs.second;
}

//have_corners：图像已有特征点， corners：用来存放额外检测到的特征点， maxCorners:图像最多特征点数量
void efficientGoodFeaturesToTrack(InputArray _image, vector<cv::Point2f>& have_corners, vector<cv::Point2f>& corners, int maxCorners, double minDistance)
{
    corners.clear();
    Mat image = _image.getMat();
    if(image.empty())
        return;

    vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, 20, true);

    if(keypoints.empty())
        return;

    cv::Mat kernal_x = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernal_y = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    vector<double> grad_x;
    vector<double> grad_y;
    grad_x = convolution(image, keypoints, kernal_x);
    grad_y = convolution(image, keypoints, kernal_y);

    //2. minMaxLoc
    vector<pair<int, double>> eigens;
    for(int i = 0; i < grad_x.size(); i++)
    {
        Eigen::Matrix2d cov;
        cov(0, 0) = grad_x[i] * grad_x[i];
        cov(0, 1) = grad_x[i] * grad_y[i];
        cov(1, 0) = grad_x[i] * grad_y[i];
        cov(1, 1) = grad_y[i] * grad_y[i];

        EigenSolver<Matrix2d> es(cov);
        Eigen::Vector2cd eig_ = es.eigenvalues();
        Vector2d eig = eig_.real();
        double eg1 = eig(0);
        double eg2 = eig(1);
        if(eg1 >= eg2)
            eigens.push_back(make_pair(i, eg1));
        else
            eigens.push_back(make_pair(i, eg2));
        
    }

    sort(eigens.begin(), eigens.end(), cmp_by_value);
    vector<cv::KeyPoint> keypoints_;
    for(int i = 0; i < eigens.size(); i++)
        keypoints_.push_back(keypoints[eigens[i].first]);

    if(minDistance >= 1)
    {
        int ncorners = 0;
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size -1) / cell_size;
        const int grid_height = (h + cell_size -1) / cell_size;

        std::vector<std::vector<cv::Point2f>> grid(grid_width * grid_height);

        minDistance *= minDistance;
        //push the already exist feature points into the grid
        for(int i = 0; i < have_corners.size(); i++)
        {
            int y = (int)(have_corners[i].y);
            int x = (int)(have_corners[i].x);

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            if(x_cell <= grid_width && y_cell <= grid_height)
                grid[y_cell*grid_width + x_cell].push_back(have_corners[i]);

            corners.push_back(have_corners[i]);
            ++ncorners;
        }

        for(int i = 0; i < keypoints_.size(); i++)
        {
            if(keypoints_[i].pt.y < 0 || keypoints_[i].pt.y > image.rows - 1)
                continue;
            if(keypoints_[i].pt.x < 0 || keypoints_[i].pt.x > image.cols - 1)
                continue;    
            int y = (int)(keypoints_[i].pt.y);
            int x = (int)(keypoints_[i].pt.x);

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell -1;
            int y1 = y_cell -1;
            int x2 = x_cell +1;
            int y2 = y_cell +1;

            //boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            //select feature points satisfy minDistance threshold
            for(int yy = y1; yy <= y2; yy++)
            {
                for(int xx = x1; xx <= x2; xx++)
                {
                    std::vector<cv::Point2f>& m = grid[yy*grid_width+xx];

                    if(m.size())
                    {
                        for(int j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if(dx*dx + dy*dy < minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if(good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if(maxCorners > 0 && (int)ncorners == maxCorners)
                    break;
            }
        }

    }
    else
    {
        return;
    }

    if(have_corners.size() != 0)
        corners.erase(corners.begin(), corners.end()+have_corners.size());
    
}

int main()
{
	vector<cv::Point2f> forw_pts, n_pts;
	cv::Mat img_ = cv::imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat img = cv::imread("test.jpg", 0);
	cv::Mat mask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
	system_clock::time_point t1 = std::chrono::system_clock::now();
	cv::goodFeaturesToTrack(img, n_pts, 200, 0.01, 30, mask);
	system_clock::time_point t2 = std::chrono::system_clock::now();
	std::cout <<"goodFeaturesToTrack() time " << fixed << setprecision(6) << duration_cast<milliseconds>(t2 - t1).count() << " ms" << std::endl;
        for(int i = 0; i < n_pts.size(); i++)
	{
	    cv::circle(img_, n_pts[i], 2, Scalar(0, 0, 255), 2);
        }	    
	system_clock::time_point t3 = std::chrono::system_clock::now();
	efficientGoodFeaturesToTrack(img, forw_pts, n_pts, 200, 30);
	system_clock::time_point t4 = std::chrono::system_clock::now();
	std::cout << "efficientGoodFeaturesToTrack time " << fixed << setprecision(6) << duration_cast<milliseconds>(t4 - t3).count() << " ms" << std::endl;
        for(int i = 0; i < n_pts.size(); i++)
	{
	    cv::circle(img_, n_pts[i], 2, Scalar(255, 0, 0), 2);
	}
	cv::imwrite("cmp.jpg", img_);

	return 0;








}	
