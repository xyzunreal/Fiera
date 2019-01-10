#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct conv_layer_bin_t
{
	layer_type type = layer_type::conv_bin;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_bin_t in_bin;
	tensor_bin_t out_bin;
	tensor_t<float> filters; //std::vector<tensor_t<float>> filters;
	tensor_bin_t filters_bin; //vector<tensor_bin_t> filters_bin;
	tensor_t<gradient_t> filter_grads; //std::vector<tensor_t<gradient_t>> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;
	vector<float> alpha;
	
	conv_layer_bin_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size )
		:
		grads_in(in_size.m, in_size.x, in_size.y, in_size.z),
		in(in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		in_bin(in_size.m, in_size.x, in_size.y, in_size.z),
		out_bin(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		filters(number_filters, extend_filter, extend_filter, in_size.z),
		filter_grads(number_filters, extend_filter, extend_filter, in_size.z),
		filters_bin(number_filters, extend_filter, extend_filter, in_size.z)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ )
		{
			// tensor_t<float> t( extend_filter, extend_filter, in_size.z );
			// tensor_bin_t tb(extend_filter, extend_filter, in_size.z);
			int maxval = extend_filter * extend_filter * in_size.z;

			for ( int i = 0; i < extend_filter; i++ ){
			
				for ( int j = 0; j < extend_filter; j++ ){
					
					for ( int z = 0; z < in_size.z; z++ ){
						 //initialization of floating weights 
						//t( i, j, z ) = (1.0f / maxval * (rand()-rand()) / float( RAND_MAX ));
						 /**************temporary*************/
						 filters(a,i,j,z) = pow(-1,i^j)*2+i+j-3;
						// initialization of binary weights
						filters_bin.data[filters_bin(a,i,j,z)] = 0;
					}
				}
			}
			// filters.push_back( t );
			// filters_bin.push_back(tb);
		}

		cout<<"******weights for conv_bin********\n"<<endl;
		print_tensor(filters));
		
		// for ( int i = 0; i < number_filters; i++ )
		// {
		// 	tensor_t<gradient_t> t( extend_filter, extend_filter, in_size.z );
		// 	filter_grads.push_back( t );
		// }

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

	int normalize_range( float f, int max, bool lim_min )
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
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size() - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}
	
	
	void binarize(){
		
		// binarizes weights
		for(int filter = 0; filter<filters.size.m; filter++){
			
			// tensor_t<float> &tf = filters[filter];
			// tensor_bin_t &tb = filters_bin[filter];
			
			for(int x=0; x< filters_bin[filter].size.x; x++){
				for(int y=0; y< filters_bin[filter].size.y; y++){
					for(int z=0; z< filters_bin[filter].size.z; z++){
						// ************************ remember always take var.data[var(x,y,z)] *****************************
						if(filters(filter,x,y,z) >= 0) filters_bin.data[filters_bin(filter,x,y,z)] = 1;
						else filters_bin.data[filters_bin(filter,x,y,z)] = 0;
					}
				}
			}
		}

		cout<<"************ binarized weights **********\n";
		print_tensor_bin(filters_bin);
		
		//binarizes in 
		for(int example = 0; example<in.size.m; example++)
			for(int x=0; x<in.size.x; x++){
				for(int y=0; y<in.size.y; y++){
					for(int z=0; z<in.size.z; z++){
						in_bin.data[in_bin(example,x,y,z)] = in(example,x,y,z)>=0 ? 1 : 0;
						
					}
				}
			}
		cout<<"******binarize input**************"<<endl;
		print_tensor_bin(in_bin);
		
	}
	
	void cal_mean(){
		//cout<<filters.size()<<' '<<filters[0].size.x<<' '<<filters[0].size.y<<' '<<filters[0].size.z<<endl;
		
		alpha.resize(filters.size());
		
		for(int filter = 0; filter<filters.size.m; filter++){
			float sum = 0;
			// tensor_t<float> &tf = filters[filter];
			for(int x=0; x<tf.size.x; x++)
				for(int y=0; y<tf.size.y; y++)
					for(int z=0; z<tf.size.z; z++){
						sum += filters(filter,x,y,z);
				}
			alpha[filter] = sum/(filters.size.x*filters.size.y*filters.size.z);
		}
		
		cout<<"*******mean for weights*********\n";
		cout<<alpha[0]<<endl;
		
		//~ return sum/(filters.size()*);
	}
	
	void activate()
	{
		//initialize alpha for xornet
		cal_mean();
		//binarize filters and in 
		binarize();

		//binarize convolution :)

		for(int example = 0; example<in.size.m; example++)
		
			for ( int filter = 0; filter < filters_bin.size.m; filter++ )
			{
				// tensor_bin_t &filter_data = filters_bin[filter];
				for ( int x = 0; x < out.size.x; x++ )
				{
					for ( int y = 0; y < out.size.y; y++ )
					{
						point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0;
						for ( int i = 0; i < extend_filter; i++ )
							for ( int j = 0; j < extend_filter; j++ )
								for ( int z = 0; z < in.size.z; z++ )
								{
									bool f = filters_bin.data[filter_bin(filter, i, j, z )];
									bool v = in_bin.data[in_bin(example, mapped.x + i, mapped.y + j, z )];
									sum += (!(f^v));
								}
						out(example, x, y, filter ) = (2*sum - extend_filter*extend_filter*in.size.z);
						
						out(example, x,y,filter) *= alpha[filter];
						
					}
				}
			}
		cout<<"*********output for conv_bin*********\n";
		print_tensor(out);
	}
	
	
	
	void fix_weights()
	// {
	// 	for ( int a = 0; a < filters.size(); a++ )
	// 		for ( int i = 0; i < extend_filter; i++ )
	// 			for ( int j = 0; j < extend_filter; j++ )
	// 				for ( int z = 0; z < in.size.z; z++ )
	// 				{
	// 					float& w = filters[a].get( i, j, z );
	// 					gradient_t& grad = filter_grads[a].get( i, j, z );
	// 					w = update_weight( w, grad );
	// 					update_gradient( grad );
	// 				}
	}

	void calc_grads( tensor_t<float>& grad_next_layer, bool ismini = false)
	{

		// for ( int k = 0; k < filter_grads.size(); k++ )
		// {
		// 	for ( int i = 0; i < extend_filter; i++ )
		// 		for ( int j = 0; j < extend_filter; j++ )
		// 			for ( int z = 0; z < in.size.z; z++ )
		// 				filter_grads[k].get( i, j, z ).grad = 0;
		// }

		// for ( int x = 0; x < in.size.x; x++ )
		// {
		// 	for ( int y = 0; y < in.size.y; y++ )
		// 	{
		// 		range_t rn = map_to_output( x, y );
		// 		for ( int z = 0; z < in.size.z; z++ )
		// 		{
		// 			float sum_error = 0;
		// 			for ( int i = rn.min_x; i <= rn.max_x; i++ )
		// 			{
		// 				int minx = i * stride;
		// 				for ( int j = rn.min_y; j <= rn.max_y; j++ )
		// 				{
		// 					int miny = j * stride;
		// 					for ( int k = rn.min_z; k <= rn.max_z; k++ )
		// 					{
		// 						int w_applied = filters[k].get( x - minx, y - miny, z );
		// 						sum_error += w_applied * grad_next_layer( i, j, k );
		// 						filter_grads[k].get( x - minx, y - miny, z ).grad += in( x, y, z ) * grad_next_layer( i, j, k );
		// 					}
		// 				}
		// 			}
		// 			grads_in( x, y, z ) = sum_error;
		// 		}
		// 	}
		// }
	}
};
#pragma pack(pop)
