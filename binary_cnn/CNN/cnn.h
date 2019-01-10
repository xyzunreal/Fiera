#pragma once
#include "tensor_t.h"
#include "tensor_bin_t.h"
#include "optimization_method.h"
#include "fc_layer.h"
#include "pool_layer_t.h"
#include "prelu_layer_t.h"
#include "conv_layer_t.h"
#include "dropout_layer_t.h"
#include "conv_layer_bin_t.h"
#include "fc_layer_bin.h"
#include "scale_layer_t.h"
#include "softmax_layer_t.h"

using namespace std;


static void calc_grads( layer_t* layer, tensor_t<float>& grad_next_layer )
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		//to be checked for binary
		//~ case layer_type::scale:
			//~ ((scale_layer_t*)layer)->calc_grads( grad_next_layer );
			//~ return;
		//to be checked for binary
		case layer_type::conv_bin:
			((conv_layer_bin_t*)layer)->calc_grads( grad_next_layer );
			return;
		// to be checked
		case layer_type::softmax:
			((softmax_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		// to be checked for binary
		case layer_type::fc_bin:
			((fc_layer_bin_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::prelu:
			((prelu_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->calc_grads( grad_next_layer );
			return;
		// case layer_type::dropout_layer:
		// 	((dropout_layer_t*)layer)->calc_grads( grad_next_layer );
		// 	return;
		default:
			assert( false );
	}
}

static void fix_weights( layer_t* layer )
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->fix_weights();
			return;
		//to be checked for binary
		case layer_type::scale:
			((scale_layer_t*)layer)->fix_weights();
			return;
		// to be checked
		case layer_type::softmax:
			((softmax_layer_t*)layer)->fix_weights();
			return;
		// to be checked for binary convolution
		case layer_type::conv_bin:
			((conv_layer_bin_t*)layer)->fix_weights();
			return;
		case layer_type::prelu:
			((prelu_layer_t*)layer)->fix_weights();
			return;
		//to be checked for binary fc
		case layer_type::fc_bin:
			((fc_layer_bin_t*)layer)->fix_weights();
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->fix_weights();
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->fix_weights();
			return;
		// case layer_type::dropout_layer:
		// 	((dropout_layer_t*)layer)->fix_weights();
		// 	return;
		default:
			assert( false );
	}
}

static void activate( layer_t* layer, tensor_t<float>& in )
{
	switch ( layer->type )
	{
		case layer_type::conv:
			((conv_layer_t*)layer)->activate( in );
			return;
		// to be checked for binary convolution
		case layer_type::scale:
			((scale_layer_t*)layer)->activate( in );
			return;
		// to be checked for binary convolution
		case layer_type::conv_bin:
			((conv_layer_bin_t*)layer)->activate( in );
			return;
		case layer_type::softmax:
			((softmax_layer_t*)layer)->activate( in );
			return;
		case layer_type::prelu:
			((prelu_layer_t*)layer)->activate( in );
			return;
		// to be checked for binary fc
		case layer_type::fc_bin:
			((fc_layer_bin_t*)layer)->activate( in );
			return;
		case layer_type::fc:
			((fc_layer_t*)layer)->activate( in );
			return;
		case layer_type::pool:
			((pool_layer_t*)layer)->activate( in );
			return;
		// case layer_type::dropout_layer:
		// 	((dropout_layer_t*)layer)->activate( in );
		// 	return;
		default:
			assert( false );
	}
}
