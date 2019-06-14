#pragma once
#include "tensor_t.h"
#include "data.h"

#pragma pack(push, 1)

struct Dataset
{
	Data train;
	Data test;
	Data validation;

	Dataset(int ntrain, int ntest, int nval, int nout, int img_w, int img_h, int img_c){
		train.images = tensor_t<float>(ntrain, img_w, img_h, img_c);
		train.labels = tensor_t<float>(ntrain, nout, 1, 1);
		test.images = tensor_t<float>(ntest, img_w, img_h, img_c);
		test.labels = tensor_t<float>(ntest, nout, 1, 1);
		validation.images = tensor_t<float>(nval, img_w, img_h, img_c);
		validation.labels = tensor_t<float>(nval, nout, 1, 1);
	}
	void operator = (Dataset data){
		train = data.train;
		test = data.test;
		validation = data.validation;
	}
};

#pragma pack(pop)