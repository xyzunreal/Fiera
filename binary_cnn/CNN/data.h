/*! Custom Class to hold data as 'images' and 'labels'*/

#pragma once
#include "tensor_t.h"

#pragma pack(push, 1)

struct Data{
    tensor_t<float> images;
    tensor_t<float> labels;

    void operator = (Data data){
		images = data.images;
		labels = data.labels;
	}
};

#pragma pack(pop)