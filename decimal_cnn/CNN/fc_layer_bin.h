/* TODO: Implement `binarize_layer` and `binarize_weights` function when tensor_binary_t.h gets completed.
         Change `activate` and `calc_grade` functions accordingly.
         Recheck Constructor after completion of tensor_binary.h
*/

#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"

#pragma pack(push, 1)
struct fc_layer_t
{
	layer_type type = layer_type::fc_bin;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
    tensor_bin_t in_bin;
    tensor_bin_t out_bin;
    float alpha;

	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_t( tdsize in_size, int out_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
        in_bin(),
		out( out_size, 1, 1 ),
        out_bin(),
		grads_in( in_size.x, in_size.y, in_size.z ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		input = std::vector<float>( out_size );
		gradients = std::vector<gradient_t>( out_size );


		int maxval = in_size.x * in_size.y * in_size.z;

		// WEIGHT INITIALIZATION
		
		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
				weights( h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
		// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

    void binarize_layer()
    {
    }

    void binarize_weights()
    {

    }

    void calculate_alpha(int m=1)
    {
        // CAN USE MULTIPLE BINARISATION BY CHANGING `calculate_alpha` and `multiply_by_alpha` FUNCTIONS
        float a = 0;
        for (int i = 0; i < in.size.x; i++)
            for (int j = 0; j < in.size.y; j++){
                for (int z = 0; z < in.size.z; z++)
                {
                    a += in(i, j, z);
                }
            int m = in.size.z * in.size.y * in.size.x;      // NO OF ELEMENTS IN `in`
            alpha = a / m;
            }
    }

    void multiply_by_alpha(int m=1)
    {
        for (int n = 0; n < out.size.x; n++)
        {
            out(n,1,1) = out_bin(n,1,1) * alpha;
        }
    }   

	int map( point_t d )
	// `tensor_t` saves data in 1D format. `map` maps 3D point to 1D tensor.
	{
		return d.z * (in.size.x * in.size.y) +
			d.y * (in.size.x) +
			d.x;
	}

	void activate()
	/* 
	 * `activate` activates (forward propogate) the fc layer.
	 * It saves the result after propogation in `out` variable.
	 */
	{

        binarize_layer();
        binarize_weights();
        
        // TODO: Multiply weights and in_bin according to structure of tensor_binary

        multiply_by_alpha();

        // OLD CODE - TO BE USED AS REFERENCE
		// for ( int n = 0; n < out.size.x; n++ )
		// {
		// 	float inputv = 0;

		// 	for ( int i = 0; i < in.size.x; i++ )
		// 		for ( int j = 0; j < in.size.y; j++ )
		// 			for ( int z = 0; z < in.size.z; z++ )
		// 			{
		// 				int m = map( { i, j, z } );
		// 				inputv += in( i, j, z ) * weights( m, n, 0 );
		// 			}

		// 	input[n] = inputv;
		// 	out( n, 0, 0 ) = inputv;
		// }
	}

	void fix_weights()
	{
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						float& w = weights( m, n, 0 );
						w = update_weight( w, grad, in( i, j, z ) );
					}

			update_gradient( grad );
		}
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	
	// Calculates backward propogation and saves result in `grads_in`. 
	

	{
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( input[n] );

			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						grads_in( i, j, z ) += grad.grad * weights( m, n, 0 );
					}
		}
	}
};
#pragma pack(pop)
