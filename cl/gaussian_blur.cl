__kernel void gaussian_blur(
        __global uchar * image,
        //__constant float * mask,
        __global uchar4 * blurredImage,
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
    uchar4 sum = 0;

    for (int i = -r; i <= r; i++) {
	for (int j = -r; j <= r; j++) {
	    sum = sum + image[imgPos - imageWidth * j * 3 + i * 3] / (maskSize * maskSize);
	}
    }
    //blurredImage[blurPos] = sum;

    blurredImage[blurPos].x = image[imgPos];
    blurredImage[blurPos].y = image[imgPos + 1];
    blurredImage[blurPos].z = image[imgPos + 2];
    blurredImage[blurPos].w = 100;
}
