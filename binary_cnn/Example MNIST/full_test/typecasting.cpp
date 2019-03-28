#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

float train( vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected, float learning_rate = 0.1 )
{
    
    for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
        else
			activate( layers[i], layers[i - 1]->out );

	}

    float l1 = cross_entropy(layers[layers.size()-1]->out, expected);

    for ( int i = layers.size() - 1; i >= 0; i-- )
	{
		if ( i == layers.size() - 1 )
			calc_grads( layers[i], expected);
		else
			calc_grads( layers[i], layers[i + 1]->grads_in );
	}

    for ( int i = 0; i < layers.size(); i++ )
	{
		fix_weights( layers[i], learning_rate);
	}

    return l1;

}
int main()
{
	// vector<layer_t*> layers;
    tensor_t<float> temp_in(2, 3,3,2), predict(2,4,1,1);
    
   std::vector<std::vector<std::vector<std::vector<float> > > > vect=
          {{{{-0.0145, -0.3839, -2.9662},
          {-1.0606, -0.3090,  0.9343},
          {-0.3821, -1.1669, -0.4375}},

         {{-2.1085,  1.1450, -0.3822},
          {-0.3553,  0.7542,  0.6901},
          {-0.1443, -0.5146,  0.8005}}},


        {{{-1.2432, -1.7178,  1.7518},
          { 0.9796,  0.4105,  1.7675},
          {-0.0832,  0.5087,  1.1178}},

         {{ 1.1286,  0.5013,  1.4206},
          { 1.1542, -1.5366, -0.5577},
          {-0.4383,  1.1572,  0.0889}}}};
    temp_in.from_vector(vect);

    for(int i=0; i<4; i++) 
        for(int img=0;img<2;img++) 
            predict(img,i,0,0) = 0;
    predict(0,3,0,0) = 1;
    predict(1 ,1, 0, 0) = 1;


	conv_layer_t * layer1 = new conv_layer_t(1,2,2,temp_in.size,false,false);

    prelu_layer_t * layer2 = new prelu_layer_t( layer1->out.size, false, false);

    fc_layer_t * layer8 = new fc_layer_t(layer2->out.size, 4);
     
    softmax_layer_t * layer11 = new softmax_layer_t(layer8->out.size,false);
   
    
    
    vector<float> cost_vec;
    // cost_vec.push_back(0);
    
    float learning_rate = 0.1;

    vector<layer_t *> layers;
    
    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer8);
    layers.push_back((layer_t *) layer11);

    for(int epoch = 0; epoch<100; epoch++){
        float cost = train(layers, temp_in, predict, learning_rate);
        cost_vec.push_back(cost);
    }

    cout<<"**************cost*************\n";
    for(int i=0; i<cost_vec.size(); i++){
        cout<<cost_vec[i]<<endl;
    }
    return 0;
}
