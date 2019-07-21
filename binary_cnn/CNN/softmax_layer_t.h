
/*! Softmax layer 
	It follows:
	  output(i) = exp(xi)/sum(exp(x1)+...+exp(xn)) 
*/

//TODO: Adding debug flags to ifdef


#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"

#pragma pack(push, 1)
struct softmax_layer_t
{
	layer_type type = layer_type::softmax;
	tensor_t<float> in;
	tdsize in_size, out_size;
	tensor_t<float> out;
	bool to_normalize;
	bool debug, clip_gradients_flag;	
	softmax_layer_t( tdsize in_size, bool clip_gradients_flag = true, bool debug_flag = false)
		/**
		* 
		* Parameters
		* ----------
		* in_size : (int m, int x, int y, int z)
		* 		Size of input matrix.
		*
		* clip_gradients_flag : bool
		* 		Whether gradients have to be clipped or not
		* 
		* debug_flag : bool
		* 		Whether to print variables for debugging purpose
		*
		**/
	{

		this->in_size = in_size;
		this->out_size = in_size;
		this->to_normalize = to_normalize;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}
	
	
	tensor_t<float> activate( tensor_t<float> in, bool train = true)
	//  `activate` FORWARD PROPOGATES AND RETURNS THE RESULT.	
	{	
		/** 
     	* @param 'in': input tensor for size expected (m, 10, 1, 1) 
     	*/

		tensor_t<float> out( in.size.m, in.size.x, 1, 1 );

		float temp1, sum = 0;
		
		for ( int tm = 0; tm < in.size.m; tm++ ){
			sum = 0;
			for ( int i = 0; i < in.size.x; i++ )
			{
				temp1= in(tm, i, 0, 0);
				sum += max(exp(temp1), float(1e-7));
			}
		
			for ( int i = 0; i < in.size.x; i++ )
				out(tm, i, 0, 0) = max(exp(in(tm, i, 0, 0)), float(1e-7))/sum;	
		}

		// to save inference time
		if (train) this->out = out;
		
		return out;
	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	tensor_t<float> calc_grads( tensor_t<float>& grad_next_layer )
	//  `calc_grads` BACKWARD PROPOGATES AND RETURNS THE RESULT.	
	{
		
		float m = grad_next_layer.size.m;
		
		tensor_t<float> grads_in( m, in_size.x, 1, 1 );

		for(int e = 0; e < m; e++){
			int idx;
			for(int i=0; i<out_size.x; i++){
				if(int(grad_next_layer(e,i,0,0)) == 1){
					idx = i;
					grads_in(e,i,0,0) = -(1-out(e,i,0,0))/m;
				}
			}
			for(int i=0; i<out_size.x; i++){
		 		if(idx!=i){
					grads_in(e,i,0,0) = ((out(e,i,0,0))/m);
				}
			}	
		}

		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "softmax" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "to_normalize", to_normalize },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}	

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json j = {{"type", "softmax"}};
		file << j;
		file.close();
	}

	void load_layer_weight(string fileName){

	}
	void print_layer(){
		cout << "\n\n Softmax Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)