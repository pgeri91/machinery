#include "machinery/algorithm/BackgroundSubtractionCuda.h"

namespace kerberos
{
    void BackgroundSubtractionCuda::setup(const StringMap & settings)
    {
        Algorithm::setup(settings);
        int erode = std::atoi(settings.at("algorithms.BackgroundSubtractionCuda.erode").c_str());
        int dilate = std::atoi(settings.at("algorithms.BackgroundSubtractionCuda.dilate").c_str());
        setErodeKernel(erode, erode);
        setDilateKernel(dilate, dilate);

        m_subtractor = cv::cuda::createBackgroundSubtractorMOG2();
        //cv::namedWindow("test", cv::WINDOW_NORMAL);

        std::string shadows = settings.at("algorithms.BackgroundSubtractionCuda.shadows");
        int history = std::atoi(settings.at("algorithms.BackgroundSubtractionCuda.history").c_str());
        int nmixtures = std::atoi(settings.at("algorithms.BackgroundSubtractionCuda.nmixtures").c_str());
        double ratio = std::atof(settings.at("algorithms.BackgroundSubtractionCuda.ratio").c_str());
        int threshold = std::atoi(settings.at("algorithms.BackgroundSubtractionCuda.threshold").c_str());
        m_subtractor->setDetectShadows((shadows == "true"));
        m_subtractor->setHistory(history);
        m_subtractor->setNMixtures(nmixtures);
        m_subtractor->setBackgroundRatio(ratio);
        m_subtractor->setVarThreshold(threshold);
        m_subtractor->setVarThresholdGen(threshold);
    }

    // ---------------------------------------------
    // Convert all images (except last one) to gray

    void BackgroundSubtractionCuda::initialize(ImageVector & images)
    {
        for(int i = 0; i < images.size()-1; i++)
        {
            cv::cuda::GpuMat gpuImage;
            cv::cuda::GpuMat gpuBackground;
            gpuImage.upload(images[i]->getImage());
            gpuBackground.upload(m_backgroud.getImage());
            m_subtractor->apply(gpuImage, gpuBackground);
            cv::Mat newBackground(gpuBackground);
            m_backgroud.setImage(newBackground);
        }
    }

    Image BackgroundSubtractionCuda::evaluate(ImageVector & images, JSON & data)
    {
        // -----------
        // Calculate
        cv::cuda::GpuMat gpuImage;
        cv::cuda::GpuMat gpuBackground;
        gpuImage.upload(images[2]->getImage());
        gpuBackground.upload(m_backgroud.getImage());
        m_subtractor->apply(gpuImage, gpuBackground);
        cv::Mat newBackground(gpuBackground);
        m_backgroud.setImage(newBackground);
        m_backgroud.erode(m_erodeKernel);
        m_backgroud.dilate(m_dilateKernel);
        //cv::imshow("test", m_backgroud.getImage());
        //cv::waitKey(1);
        return m_backgroud;
    }

    void BackgroundSubtractionCuda::setErodeKernel(int width, int height)
    {
        m_erodeKernel.setImage(Image::createKernel(width, height));
    }

    void BackgroundSubtractionCuda::setDilateKernel(int width, int height)
    {
        m_dilateKernel.setImage(Image::createKernel(width, height));
    }

    void BackgroundSubtractionCuda::setThreshold(int threshold)
    {
        m_threshold = threshold;
    }
}
