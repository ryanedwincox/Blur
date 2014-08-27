__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 
__kernel void gaussian_blur(
        __global float * image,
        //__constant float * mask,
        __global float * blurredImage,
	int imageWidth,
	int imageHeight
        //__private int maskSize
    ) {
 
    //const int2 pos = {get_global_id(0), get_global_id(1)};
 
    // Collect neighbor values and multiply with Gaussian
    //float sum = 0.0f;
    //for(int a = -maskSize; a < maskSize+1; a++) {
    //    for(int b = -maskSize; b < maskSize+1; b++) {
    //        sum += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]
    //                   *image[pos + (int2)(a,b)];
		//*read_imagef(image, sampler, pos + (int2)(a,b)).x;
    //    }
    //}
 
    //blurredImage[pos.x+pos.y*get_global_size(0)] = sum;

    const int xpos = get_global_id(0);
    const int ypos = get_global_id(1); 

    blurredImage[ypos * imageWidth + xpos] = image[ypos * imageWidth + xpos];

}
