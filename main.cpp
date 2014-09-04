//color
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
#define MAX_LOG_SIZE (0x100000)

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
    image = cv::imread("/home/pierre/Documents/tutorials/blur/images/Lenna.png", CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }

    size_t imageWidth = image.cols;
    size_t imageHeight = image.rows;
    size_t imageSize = imageHeight * imageWidth;
    std::cout << "image width: " << imageWidth << "\n";
    std::cout << "image height: " << imageHeight << "\n";

    unsigned char newData [imageSize * 3];

    // get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

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
    std::cout << "context error: " << err << "\n";

    // initialize
    FILE* programHandle;
    char *programBuffer;
    size_t programSize;
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

    // build program
    const char* buildOptions = "";
    cl_int warn = clBuildProgram(program, deviceCount, devices, buildOptions, NULL, NULL);
    std::cout << "program error: " << warn << "\n";

    // create the log string and show it to the user. Then quit
    char buildLog[MAX_LOG_SIZE];
    err = clGetProgramBuildInfo(program,
                          devices[0],
                          CL_PROGRAM_BUILD_LOG,
                          MAX_LOG_SIZE,
                          &buildLog,
                          NULL);
    printf("**BUILD LOG**\n%s",buildLog);
    std::cout << "clGetProgramBuildInfo error: " << err << "\n";

    //create queue to which we will push commands for the device.
    cl_command_queue queue;
    queue = clCreateCommandQueue(context,devices[0],0,&err);
    std::cout << "command queue error: " << err << "\n";

    // Create an OpenCL Image / texture and transfer data to the device
    cl_mem clImage = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY,
                                    imageSize * 3,
                                    NULL,
                                    &err);
    std::cout << "clImage error: " << err << "\n";

    // Create an OpenCL Image for the result
    cl_mem clResult = clCreateBuffer(context,
                                     CL_MEM_WRITE_ONLY,
                                     imageSize * 3,
                                     NULL,
                                     &err);
    std::cout << "clResult error: " << err << "\n";

    // Create Gaussian mask
    int maskSize;
    float * mask = createBlurMask(10.0f, &maskSize);

    // Create buffer for mask
    cl_mem clMask = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)*(maskSize*2+1)*(maskSize*2+1),
                                   mask,
                                   &err);
    std::cout << "clMask error: " << err << "\n";

    // create Gaussian kernel
    cl_kernel gaussianBlur = clCreateKernel(program, "gaussian_blur", &err);
    std::cout << "cl_kernel error: " << err << "\n";

    // set kernel arguments
    err = clSetKernelArg(gaussianBlur, 0, sizeof(cl_mem), (void *)&clImage);
    std::cout << "kernel arg 0 error: " << err << "\n";
//    clSetKernelArg(gaussianBlur, 1, sizeof(cl_mem), &clMask);
    err = clSetKernelArg(gaussianBlur, 1, sizeof(cl_mem), (void *)&clResult);
    std::cout << "kernel arg 1 error: " << err << "\n";
    err = clSetKernelArg(gaussianBlur, 2, sizeof(int), &imageWidth);
    std::cout << "kernel arg 2 error: " << err << "\n";
    err = clSetKernelArg(gaussianBlur, 3, sizeof(int), &imageHeight);
    std::cout << "kernel arg 3 error: " << err << "\n";
//    clSetKernelArg(gaussianBlur, 5, sizeof(cl_int), &maskSize);

    // load image to device
    err = clEnqueueWriteBuffer(queue,
                               clImage,
                               CL_TRUE,
                               0,
                               imageSize * 3,
                               (void*) &image.data[0],
                               0,
                               NULL,
                               NULL);
    std::cout << "enqueueWriteImage error: " << err << "\n";

    // Set local and global workgroup sizes
    size_t localws[2] = {16,16};
    size_t globalws[2] = {imageWidth, imageHeight};

    // Run Gaussian kernel
    err = clEnqueueNDRangeKernel(queue,
                                 gaussianBlur,
                                 2,
                                 NULL,
                                 globalws,
                                 localws,
                                 0,
                                 NULL,
                                 NULL);
    std::cout << "clEnqueueNDRangeKernel error: " << err << "\n";

    // Transfer image back to host
    err = clEnqueueReadBuffer(queue,
                              clResult,
                              CL_TRUE,
                              0,
                              imageSize * 3,
                              (void*) newData,
                              NULL,
                              NULL,
                              NULL);
    std::cout << "enqueueReadImage error: " << err << "\n";

    cv::Mat newImage = cv::Mat(cv::Size(imageWidth,imageHeight), CV_8UC3, newData);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow("Original Image", image);                   // Show our image inside it.

    cv::namedWindow("Blured Image", cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow("Blured Image", newImage);            // Show our image inside it.

    std::cout << "finish";

    cv::waitKey(0);
}
