__kernel void gaussian_blur(
        __global uchar * image, //bgr
        //__constant float * mask,
        __global uchar4 * blurredImage, //bgra
	int imageWidth,
	int imageHeight,
        __private int maskSize
    ) {
    int xpos = get_global_id(0);
    int ypos = get_global_id(1); 

    int r = maskSize / 2;

    clamp(xpos, r, imageWidth - r);
    clamp(ypos, r, imageHeight - r);

    int blurPos = ypos * imageWidth + xpos;
    int imgPos = blurPos * 3;
    uchar sum0 = 0;
    uchar sum1 = 0;
    uchar sum2 = 0;

    for (int i = -r; i <= r; i++) {
	for (int j = -r; j <= r; j++) {
	    sum0 = sum0 + image[imgPos - imageWidth * j * 3 + i * 3] / (maskSize * maskSize);
	    sum1 = sum1 + image[imgPos - imageWidth * j * 3 + i * 3 + 1] / (maskSize * maskSize);
	    sum2 = sum2 + image[imgPos - imageWidth * j * 3 + i * 3 + 2] / (maskSize * maskSize);
	}
    }
    blurredImage[blurPos].x = sum0;
    blurredImage[blurPos].y = sum1;
    blurredImage[blurPos].z = sum2;
    blurredImage[blurPos].w = 255;
}
