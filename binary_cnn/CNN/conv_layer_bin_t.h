#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct conv_layer_bin_t
{
	layer_type type = layer_type::conv_bin;
	tensor_t<double> grads_in;
	tensor_t<double> in;
	tensor_t<double> out;
	tensor_bin_t in_bin; 		// 1st BINARIZATION (h1)
	tensor_bin_t in_bin2;		// 2nd BINARIZATION (h2)
	tensor_t<double> al_b; 			// α1 * h1 + α2 * h2
	tensor_bin_t out_bin;
	tensor_t<double> filters; 
	tensor_bin_t filters_bin; 
	tensor_t<gradient_t> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;
	vector<double> alpha;
	vector<double> alpha2;
	bool debug,clip_gradients_flag; 	
	conv_layer_bin_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size ,
	bool clip_gradients_flag = true, bool debug_flag = false)
		:
		grads_in(in_size.m, in_size.x, in_size.y, in_size.z),
		in(in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		in_bin(in_size.m, in_size.x, in_size.y, in_size.z),
		in_bin2(in_size.m, in_size.x, in_size.y, in_size.z),
		al_b(in_size.m, in_size.x, in_size.y, in_size.z),
		out_bin(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		filters(number_filters, extend_filter, extend_filter, in_size.z),
		filter_grads(number_filters, extend_filter, extend_filter, in_size.z),
		filters_bin(number_filters, extend_filter, extend_filter, in_size.z)

	{
		this->debug = debug_flag;
		this->stride = stride;
		this->extend_filter = extend_filter;
		this->clip_gradients_flag = clip_gradients_flag;
		assert( (double( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (double( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ ){
			int maxval = extend_filter * extend_filter * in_size.z;
			for ( int i = 0; i < extend_filter; i++ ){
				for ( int j = 0; j < extend_filter; j++ ){
					for ( int z = 0; z < in_size.z; z++ ){
						 //initialization of floating weights 
						filters(a,i, j, z ) =  (1.0f * (rand()-rand())) / double( RAND_MAX );

						// initialization of binary weights
						filters_bin.data[filters_bin(a,i,j,z)] = 0;
					}
				}
			}
		}
		if(debug)
		{
			cout<<"\n******weights for conv_bin********\n"<<endl;
			print_tensor(filters);
		}


	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( double f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		double a = x;
		double b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size.m - 1,
		};
	}

	void activate( tensor_t<double>& in )
	{
		this->in = in;
		activate();
	}
	
	
	void binarize(){
		
		// binarizes weights
		for(int filter = 0; filter<filters.size.m; filter++){
			
			for(int x=0; x< filters.size.x; x++){
				for(int y=0; y< filters.size.y; y++){
					for(int z=0; z< filters.size.z; z++){
						if(filters(filter,x,y,z) >= 0) filters_bin.data[filters_bin(filter,x,y,z)] = 1;
						else filters_bin.data[filters_bin(filter,x,y,z)] = 0;
					}
				}
			}
		}
		if(debug)
		{
			cout<<"\n************ binarized weights **********\n";
			print_tensor_bin(filters_bin);
		}
		
		//binarizes in 
		for(int example = 0; example<in.size.m; example++)
			for(int x=0; x<in.size.x; x++){
				for(int y=0; y<in.size.y; y++){
					for(int z=0; z<in.size.z; z++){
						in_bin.data[in_bin(example,x,y,z)] = in(example,x,y,z)>=0 ? 1 : 0;
						
					}
				}
				}
		
		if(debug)
		{
			cout<<"\n******binarize input**************"<<endl;
			print_tensor_bin(in_bin);
		}
		
	}
	
	void cal_mean(){
	
		alpha.resize(in.size.m);
		alpha2.resize(in.size.m);

		// CALCULATE alpha1		
		for(int e = 0; e<in.size.m; e++){
			double sum = 0;
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						sum += abs(in(e,x,y,z));
				}
			
			alpha[e] = sum/(in.size.x*in.size.y*in.size.z);
			if(debug)
			{
				cout<<"\nalpha1 for"<<e<<"th example is "<<alpha[e]<<endl;
			}
		}

		// CALCULATE alpha2
		tensor_t<double> temp(in.size.m, in.size.x, in.size.y, in.size.z);
		for(int e = 0; e<in.size.m; e++){
			
			double sum = 0;
			
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						temp(e,x,y,z) = in(e,x,y,z) - alpha[e]*(in_bin(e,x,y,z)==1 ? double(1) : double(-1) );
						in_bin2.data[in_bin2(e,x,y,z)] = temp(e,x,y,z)>=0? 1 : 0;
						sum += abs(temp(e,x,y,z));
				}

			alpha2[e] = sum/(in.size.x*in.size.y*in.size.z);

			if(debug)
			{
				cout<<"\nalpha2 for "<<e<<"th example is "<<alpha2[e]<<endl;
			}
		}

		if(debug)
		{
			cout<<"\nin_bin2"<<endl;
			print_tensor_bin(in_bin2);
		}
		

		// CALCULATE al_b
		for (int e = 0; e < in.size.m; e++)
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						al_b(e, x, y, z) = alpha[e] * (in_bin(e, x, y, z) == 1 ? double(1) : double(-1) ) +
								alpha2[e] * (in_bin2(e, x, y, z) == 1 ? double(1) : double(-1) );
					}
		
		if(debug)
		{
			cout<<"\nal_b\n"<<endl;
			print_tensor(al_b);
		}
	}

	
	void activate()
	{
		//binarize filters and in 
		binarize();

		//initialize alpha 
		cal_mean();

		//calculating binary convolution
		for(int example = 0; example<in.size.m; example++){
		
			for ( int filter = 0; filter < filters_bin.size.m; filter++ )
				for ( int x = 0; x < out.size.x; x++ )
					for ( int y = 0; y < out.size.y; y++ ){

						point_t mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						double sum = 0, sum2 = 0;
						for ( int i = 0; i < extend_filter; i++ )
							for ( int j = 0; j < extend_filter; j++ )
								for ( int z = 0; z < in.size.z; z++ )
								{
									bool f = filters_bin.data[filters_bin(filter, i, j, z )];
									bool v = in_bin.data[in_bin(example, mapped.x + i, mapped.y + j, z )];
									bool v2 = in_bin2.data[in_bin2(example, mapped.x + i, mapped.y + j, z)];
									sum += (!(f^v));
									sum2 += (!(f^v2));
								}
						out(example, x, y, filter ) = alpha[example]*(2*sum - extend_filter*extend_filter*in.size.z);
						
						out(example, x,y,filter) += alpha2[example]*(2*sum2 - extend_filter*extend_filter*in.size.z);
						
					}
				}
			
		

		if(debug)
		{
			cout<<"*********output for conv_bin*********\n";
			print_tensor(out);
		}

	}
	
	
	
	void fix_weights(double learning_rate){
		for ( int a = 0; a < filters.size.m; a++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						double& w = filters(a, i, j, z );
						gradient_t& grad = filter_grads(a, i, j, z );
						grad.grad /= in.size.m;
						w = update_weight( w, grad, 1, true, learning_rate);
						update_gradient(grad);
					}
		if(debug)
		{
			cout<<"\n*******new weights for conv_bin*****\n";
			print_tensor(filters);
		}
	}

	void calc_grads( tensor_t<double>& grad_next_layer)
	{
		for ( int k = 0; k < filter_grads.size.m; k++ )
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads(k, i, j, z ).grad = 0;
		}

		for ( int e = 0; e < in.size.m; e++)
			for ( int x = 0; x < in.size.x; x++ )
				for ( int y = 0; y < in.size.y; y++ ){
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ ){
						
						double sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ ){
							int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ ){
								int miny = j * stride;
								for ( int k = rn.min_z; k <= rn.max_z; k++ ){
									
									filter_grads(k, x - minx, y - miny, z ).grad += al_b(e, x, y, z ) * grad_next_layer(e, i, j, k );
									clip_gradients(clip_gradients_flag, filter_grads(k, x - minx, y - miny, z ).grad);
									if(in(e,x,y,z) <= 1){
										double w_applied = (filters_bin.data[filters_bin(k, x - minx, y - miny, z )] == 1? double(1) : double(-1));
										sum_error += w_applied * grad_next_layer(e, i, j, k );
									}
									else sum_error = 0;
								}
							}
						}
						grads_in( e, x, y, z ) = sum_error;
						clip_gradients(clip_gradients_flag, grads_in(e,x,y,z));
					}
				}
		if(debug)
		{
			cout<<"*********grads filter*****************\n";
			print_tensor(filter_grads);
			cout<<"*********grads_in for conv_bin************\n";
			print_tensor(grads_in);
		}
	}
};
#pragma pack(pop)