#include <stdint.h>
#include <iostream>
#include <string>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "../Libraries/stb_image.h"

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("/home/deepak/Downloads/twitter.jpeg", &width, &height, &bpp, 3);
    cout << *rgb_image<< "hell" << width;
    stbi_image_free(rgb_image);

    return 0;
}