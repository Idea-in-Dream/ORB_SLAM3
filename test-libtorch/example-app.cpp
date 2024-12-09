#include <torch/torch.h>
#include <iostream>
#include "SPDetector.hpp"
#include "opencv2/opencv.hpp"



int main() {

  
  auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << (device == torch::kCUDA ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
  torch::Device device_ = torch::Device(device);
  
  // 加载 SuperPoint 模型
  std::string WEIGHTS_PATH = "../Weights/superpoint.pt";
  auto model = new SuperPointSLAM::SPDetector(WEIGHTS_PATH, false);


  // 读取图像
  cv::Mat origin = cv::imread("../1403636579763555584.png", cv::IMREAD_GRAYSCALE);


  // 调用 SuperPoint 检测特征点和描述符
  model->detect(origin, false);


    // 存储特征点和描述符
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    float iniThFAST = 0.1;
    int iniX = 0, maxX = 752, iniY = 0, maxY = 480;
    
    // 调用 SuperPoint 检测特征点和描述符
    // 方法一
    // model->detect(origin, keypoints, descriptors);
    // 方法二
    // model->getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, keypoints, true);
    // 方法三
    model->getKeyPoints(1000, keypoints, true);

    // 在图像上绘制特征点
    cv::Mat output_image;
    cv::drawKeypoints(origin, keypoints, output_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Feature Points", output_image);
    std::cout << "特征点数量: " << keypoints.size() << std::endl;

    cv::waitKey(0);  // 等待按键退出

}

