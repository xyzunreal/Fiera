#pragma once
#include "types.h"
#include "tensor_t.h"

#pragma pack(push, 1)
struct layer_t
{
	layer_type type;
	tensor_t<double> grads_in;
	tensor_t<double> in;
	tensor_t<double> out;
};
#pragma pack(pop)