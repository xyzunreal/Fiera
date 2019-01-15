#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

int main()
{
	vector<layer_t*> layers;
    tensor_t<float> temp_in(1, 5,5,1), predict(1,5,1,1);
	
	for(int k=0; k<1; k++)
	    for(int i=0; i<5; i++)
            for(int j=0; j<5; j++)
                temp_in(0, i,j,k) = pow(-1,i^j)*2+i+j-4;
	

    for(int i=0; i<5; i++) predict(0,i,0,0) = 0;
    predict(0,3,0,0) = 1;

    cout<<"*********input**************\n";
    print_tensor(temp_in);

	conv_layer_t * layer1 = new conv_layer_t( 1, 2, 1, temp_in.size );		// 28 * 28 * 1 -> 24 * 24 * 8
    prelu_layer_t * layer2 = new prelu_layer_t( layer1->out.size );
	conv_layer_bin_t * layer3 = new conv_layer_bin_t(1, 2, 1, layer2->out.size );				// 24 * 24 * 8 -> 12 * 12 * 8
	prelu_layer_t * layer4 = new prelu_layer_t(layer3->out.size);
    batch_norm_layer_t * layer5 = new batch_norm_layer_t(layer4->out.size);					// 4 * 4 * 16 -> 10
    fc_layer_t * layer6 = new fc_layer_t(layer5->out.size, 7);
    prelu_layer_t * layer7 = new prelu_layer_t(layer6->out.size);
    fc_layer_bin_t * layer8 = new fc_layer_bin_t(layer7->out.size,5);
    prelu_layer_t * layer9 = new prelu_layer_t(layer8->out.size);
    scale_layer_t * layer10 = new scale_layer_t(layer9->out.size);
    softmax_layer_t * layer11 = new softmax_layer_t(layer10->out.size);
    // tensor_t<float> cost(1,1,1,1); 

    // cout<<cross_entropy(layer11->out, predict)(0, 0, 0, 0);

	layers.push_back( (layer_t*)layer1 );
	layers.push_back( (layer_t*)layer2 );
	layers.push_back( (layer_t*)layer3 );
	layers.push_back( (layer_t*)layer4 );
    layers.push_back( (layer_t*)layer5 );
    layers.push_back( (layer_t*)layer6 );
    layers.push_back( (layer_t*)layer7 );
    layers.push_back( (layer_t*)layer8 );
    layers.push_back( (layer_t*)layer9 );
    layers.push_back( (layer_t*)layer10 );
    layers.push_back( (layer_t*)layer11);
   
    for ( int i = 0; i < layers.size(); i++ )
	{
        if(layers[i]->type == layer_type::softmax){
            if(layers[i-1]->type == layer_type::scale){
                // print_tensor(layers[i]->in);
                // cout<<"flag3\n";
                // print_tensor(layers[i-1]->out);
            }

            // cout<<"**************\n";
            // print_tensor(layers[i+1]->in);
        
        }
		if ( i == 0 )
			activate( layers[i], temp_in);
		else
			activate( layers[i], layers[i - 1]->out);
	} 
    return 0;
}