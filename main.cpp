//#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include </opt/AMDAPP/include/CL/cl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <cstdio>

#define MAX_SOURCE_SIZE (0x100000)

float * createBlurMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
            sum += temp;
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
        }
    }
    // Normalize the mask
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;

    return mask;
}

int main( int argc, char** argv )
{
    // Load image
    cv::Mat image;
    image = cv::imread("/home/pierre/Documents/tutorials/blur/images/Lenna.png", CV_LOAD_IMAGE_UNCHANGED);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    size_t imageWidth = image.rows;
    size_t imageHeight = image.cols;
    std::cout << "image width: " << imageWidth << "\n";
    std::cout << "image height: " << imageHeight << "\n";

    uint32_t imgSize = imageWidth * imageHeight;
    unsigned char* newData [imgSize];

//    cv::Mat gray;
//    cv::cvtColor(image, gray, CV_BGR2GRAY);

//    unsigned char im[image.rows * image.cols];

//    for (int i = 0; i < image.rows; i++) {
//        for (int j = 0; j < image.cols; j++) {
//          int val =  int( image.data[i*image.step + j] );
//          im[image.cols * i + j] = val;
//          //std::cout << "   " << val;
//        }
//    }

    cl::size_t<3> origin;
    origin[0] = 0; origin[1] = 0; origin[2] = 0;
    cl::size_t<3> region;
    region[0] = image.rows; region[1] = image.cols; region[2] = 1;

    // get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

//    // get default device of the default platform
//    std::vector<cl::Device> all_devices;
//    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
//    if(all_devices.size()==0){
//        std::cout<<" No devices found. Check OpenCL installation!\n";
//        exit(1);
//    }
//    cl::Device default_device=all_devices[0];
//    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    // get first platform
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, NULL);

    // get device count
    cl_uint deviceCount;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);

    // get device count
    cl_device_id* devices;
    devices = new cl_device_id[deviceCount];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

    // create a single context for all devices
    cl_context context = clCreateContext(NULL, deviceCount, devices, NULL, NULL, &err);
    if (err == 0)
    {
        std::cout<<"Context created\n";
    }

    // initialize
    FILE* programHandle;
    char *programBuffer; char *programLog;
    size_t programSize; size_t logSize;
    cl_program program;

    // get size of kernel source
    programHandle = fopen("/home/pierre/Documents/tutorials/blur/cl/gaussian_blur.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    // create program from buffer
    program = clCreateProgramWithSource(context, 1,
            (const char**) &programBuffer, &programSize, NULL);
    free(programBuffer);

    // build program
    const char* buildOptions = "";
    cl_int warn = clBuildProgram(program, deviceCount, devices, buildOptions, NULL, NULL);
    //std::cout<<"program warning: "<<warn<<"\n";

    if (warn == CL_SUCCESS)
    {
        std::cout<<"program created\n";
    }

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,devices[0]);

    // Create an OpenCL Image / texture and transfer data to the device
    cl::Image2D clImage = cl::Image2D(context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      cl::ImageFormat(CL_R, CL_FLOAT),
                                      image.rows,
                                      image.cols,
                                      0,
                                      NULL);

    // Create an OpenCL Image for the result
    cl::Image2D clResult = cl::Image2D(context,
                                       CL_MEM_WRITE_ONLY,
                                       cl::ImageFormat(CL_R, CL_FLOAT),
                                       image.rows,
                                       image.cols,
                                       0,
                                       NULL);

    // Create Gaussian mask
    int maskSize;
    float * mask = createBlurMask(10.0f, &maskSize);

    // Create buffer for mask and transfer it to the device
    cl::Buffer clMask = cl::Buffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)*(maskSize*2+1)*(maskSize*2+1),
                                   mask);

    // create Gaussian kernel
    cl::Kernel gaussianBlur = cl::Kernel(program, "gaussian_blur");
    gaussianBlur.setArg(0, clImage);
    gaussianBlur.setArg(1, clMask);
    gaussianBlur.setArg(2, clResult);
    gaussianBlur.setArg(3, maskSize);

    // load image to device
    queue.enqueueWriteImage(clImage,
                            CL_TRUE,
                            origin,
                            region,
                            0,0,
                            image.data);

    // Run Gaussian kernel
    queue.enqueueNDRangeKernel(gaussianBlur,
                            cl::NullRange,
                            cl::NDRange(image.rows, image.cols),
                            cl::NullRange);

    // Transfer image back to host
    queue.enqueueReadImage(clResult,
                           CL_TRUE,
                           origin,
                           region,
                           0,0,
                           newData);

    cv::Mat newImage = cv::Mat(cv::Size(image.rows,image.cols), CV_8UC1, newData);

    cv::namedWindow( "Original Image", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Original Image", image );                   // Show our image inside it.

    cv::namedWindow( "Blured Image", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Blured Image", newImage );            // Show our image inside it.

    cv::waitKey(0);
}
