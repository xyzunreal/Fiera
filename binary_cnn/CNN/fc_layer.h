#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"

#pragma pack(push, 1)
struct fc_layer_t
{
	layer_type type = layer_type::fc;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	std::vector<float> input;
	tensor_t<float> weights;
	tensor_t<gradient_t> gradients;

	fc_layer_t( tdsize in_size, int out_size )
		:
		in( in_size.m, in_size.x, in_size.y, in_size.z ),
		out( in_size.m, out_size, 1, 1 ),
		grads_in( in_size.m, in_size.x, in_size.y, in_size.z ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1, 1 ),
		gradients(in_size.m, out_size, 1, 1)

	{
		int maxval = in_size.x * in_size.y * in_size.z;

		// Weight initialization
		
		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
				weights(h,i, 0, 0 ) =  (1.0f * (rand()-rand())) / float( RAND_MAX );
		
		cout << "********weights for fc************\n";
		print_tensor(weights);
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	int map( point_t d )
	// Tensor saves data in 1D format. `map` maps 3D point to 1D tensor.
	{
		return d.m * (in.size.x * in.size.y * in.size.z) +
			d.z * (in.size.x * in.size.y) +
			d.y * (in.size.x) +
			d.x;
	}

	void activate()
	/* 
	 * `activate` activates (forward propogate) the fc layer.
	 * It saves the result after propogation in `out` variable.
	 */
	{
		for ( int e = 0; e < in.size.m; e++)
			for ( int n = 0; n < out.size.x; n++ )
			{
				float inputv = 0;

				for ( int z = 0; z < in.size.z; z++ )
					for ( int j = 0; j < in.size.y; j++ )
						for ( int i = 0; i < in.size.x; i++ )
						{
							int m = map( { 0 , i, j, z } );
							inputv += in( e, i, j, z ) * weights(m, n, 0, 0 );
						}


				out( e, n, 0, 0 ) = inputv;
		}
		
		cout<<"*******output for fc**********\n";
		print_tensor(out);
	}

	void fix_weights()
	{
		for ( int n = 0; n < out.size.x; n++ )
		{
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { 0, i, j, z } );
						float& w = weights( m, n, 0, 0 );
						gradient_t grad_sum;
						gradient_t weight_grad;
						for ( int e = 0; e < out.size.m; e++ ){			
							weight_grad = gradients(e, n, 0, 0) * in(e, i, j, z);	// d W = d A(l+1) * A(l)
							grad_sum = weight_grad + grad_sum;
						}
						grad_sum = grad_sum / out.size.m;
						w = update_weight( w, grad_sum); 
					}
			for (int e = 0; e < out.size.m; e++)
				update_gradient( gradients(e, n, 0, 0) );
		}

		cout<<"*******new weights for float fc*****\n";
		print_tensor(weights);
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	
	// Calculates backward propogation and saves result in `grads_in`. 
	

	{
		// cout<<"flag\n";
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		
		for(int e=0; e<in.size.m; e++)
			for ( int n = 0; n < out.size.x; n++ )
			{
				gradient_t& grad = gradients(e,n,0,0);
				grad.grad = grad_next_layer(e, n, 0, 0 );

				for ( int i = 0; i < in.size.x; i++ )
					for ( int j = 0; j < in.size.y; j++ )
						for ( int z = 0; z < in.size.z; z++ )
						{
							int m = map( {0, i, j, z } );
							grads_in(e, i, j, z ) += grad.grad * weights( m, n,0, 0 );
						}
			}
		
		// cout<<"*****grads_next_layer*****\n";
		// print_tensor(grad_next_layer);

		cout<<"**********grads_in for float fc***********\n";
		print_tensor(grads_in);
		

		// cout<<"weights_gre"
	}
};
#pragma pack(pop)
