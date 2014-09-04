__kernel void gaussian_blur(
        __global unsigned char * image,
        //__constant float * mask,
        __global unsigned char * blurredImage,
	int imageWidth,
	int imageHeight
        //__private int maskSize
    ) {
    const int xpos = get_global_id(0);
    const int ypos = get_global_id(1); 

    blurredImage[ypos * imageWidth + xpos] = image[ypos * imageWidth + xpos];
    //blurredImage[ypos * imageWidth + xpos] = 1.0;
}
