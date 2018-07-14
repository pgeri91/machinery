//
//  Class: BackgroundSubtraction
//  Description: An algorithm to detect motion using background subtraction.
//  Created:     17/05/2016
//  Author:      Cédric Verstraeten
//  Mail:        cedric@verstraeten.io
//  Website:     www.verstraeten.io
//
//  The copyright to the computer program(s) herein
//  is the property of Verstraeten.io, Belgium.
//  The program(s) may be used and/or copied under 
//  the CC-NC-ND license model.
//
//  https://doc.kerberos.io/license
//
/////////////////////////////////////////////////////

#ifndef __BackgroundSubtractionCuda_H_INCLUDED__   // if BackgroundSubtractionCuda.h hasn't been included yet...
#define __BackgroundSubtractionCuda_H_INCLUDED__   // #define this so the compiler knows it has been included

#include "machinery/algorithm/Algorithm.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/cudabgsegm.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace kerberos
{
    char BackgroundSubtractionCudaName[] = "BackgroundSubtractionCuda";
    class BackgroundSubtractionCuda : public AlgorithmCreator<BackgroundSubtractionCudaName, BackgroundSubtractionCuda>
    {
        private:
            int m_threshold;
            Image m_erodeKernel;
            Image m_dilateKernel;
            Image m_backgroud;
            cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> m_subtractor;
        public:
            BackgroundSubtractionCuda(){}
            void setup(const StringMap & settings);
            void setErodeKernel(int width, int height);
            void setDilateKernel(int width, int height);
            void setThreshold(int threshold);
            void initialize(ImageVector & images);
            Image evaluate(ImageVector & images, JSON & data);
    };
}
#endif
