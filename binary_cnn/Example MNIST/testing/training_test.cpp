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
    // srand(7);   
	vector<layer_t*> layers;
    tensor_t<float> temp_in(2, 3,3,2), predict(2,3,3,2);
    // Creating image for testing purpose
    // for(int img=0;img<2;img++)
    //     for(int k=0; k<1; k++)
    //         for(int i=0; i<3; i++)
    //             for(int j=0; j<3; j++)
    //                 temp_in(img, i,j,k) = 1.0f * (rand()) / float( RAND_MAX );
    //                 // temp_in(img, i, j, k) = i*j*k-10;


   std::vector<std::vector<std::vector<std::vector<float> > > > vect=
            {{{{-0.3090,  0.9343, -0.3821},
            {-1.1669,  0.3636, -0.3156},
            {-0.5321,  0.2991, -0.3553}},

            {{ 0.7542,  0.6901, -0.1443},
            { 1.6120,  1.5671, -0.9388},
            {-1.2370, -0.5824, -0.6153}}},


            {{{ 1.5496,  0.5989, -0.0832},
            { 0.5087,  1.1178,  1.1286},
            { 0.1416, -0.5458, -0.5368}},

            {{ 0.8289, -0.5577, -0.4383},
            { 1.1572,  0.0889,  0.2659},
            {-0.1907,  0.3836,  1.9126}}}};
    temp_in.from_vector(vect);


    print_tensor(temp_in);
	
    // Creating dummy prediction data
    for(int i=0; i<3; i++) 
        for(int img=0;img<2;img++) 
            predict(img,i,0,0) = 0;
    predict(0,3,0,0) = 1;
    predict(1 ,1, 0, 0) = 1;
    
    
    
    cout<<"*********input image**************\n";
    print_tensor(temp_in);

	conv_layer_t * layer1 = new conv_layer_t( 1, 2, 2, temp_in.size, true);		
    prelu_layer_t * layer2 = new prelu_layer_t( layer1->out.size ,true);

	// conv_layer_bin_t * layer3 = new conv_layer_bin_t(1, 2, 1, layer2->out.size );
	// prelu_layer_t * layer4 = new prelu_layer_t(layer3->out.size);
    // batch_norm_layer_t * layer5 = new batch_norm_layer_t(layer4->out.size);
    fc_layer_t * layer6 = new fc_layer_t(layer2->out.size, 4,true);
    prelu_layer_t * layer7 = new prelu_layer_t(layer6->out.size,true);
    // fc_layer_bin_t * layer8 = new fc_layer_bin_t(layer7->out.size,3);
    // prelu_layer_t * layer9 = new prelu_layer_t(layer8->out.size);
    // scale_layer_t * layer10 = new scale_layer_t(layer9->out.size);
    softmax_layer_t * layer11 = new softmax_layer_t(layer7->out.size,true);
    // tensor_t<float> cost(1,1,1,1); 

   /*************************************************************************hard coded code****************************************************************************/
    
    vector<float> cost_vec;
    cost_vec.push_back(0);
    float learning_rate = 0.01;

    for(int i=0; i<1; i++){
        layer1->activate(temp_in);
        layer2->activate(layer1->out);
        // layer3->activate(layer2->out);
        // layer4->activate(layer3->out);
        // layer5->activate(layer4->out);
        layer6->activate(layer2->out);
        layer7->activate(layer6->out);
        // layer8->activate(layer7->out);
        // layer9->activate(layer8->out);
        // layer10->activate(layer9->out);
        layer11->activate(layer7->out);

        
        cout<<"loss for img1 ";
        cout<<cross_entropy(layer11->out, predict)(0, 0, 0, 0);
        cout<<"loss for img 2";
        cout<<cross_entropy(layer11->out, predict)(1, 0, 0, 0);
        cost_vec.push_back(cross_entropy(layer11->out, predict)(0, 0, 0, 0));
        cost_vec.push_back(cross_entropy(layer11->out, predict)(1,0,0,0));
        // int size = 11;

        // cross_entropy(layers[size-1]->out, expected);
        
        int tm = layer11->out.size.m;
        int tx = layer11->out.size.x;

        tensor_t<float> softmax_grads(tm,tx,1,1);

        for(int i=0; i<tm; i++)
            for(int j=0; j<tx; j++)
                if(int(predict(i,j,0,0)) == 1)
                    softmax_grads(i,j,0,0) = (-1.0/layer11->out(i,j,0,0));
 
 
 

        layer11->calc_grads(softmax_grads);
        // layer10->calc_grads(layer11->grads_in);
        // layer9->calc_grads(layer10->grads_in); 
        // layer8->calc_grads(layer9->grads_in);
        layer7->calc_grads(layer11->grads_in);
        layer6->calc_grads(layer7->grads_in);
        // layer5->calc_grads(layer6->grads_in);
        // layer4->calc_grads(layer5->grads_in);
        // layer3->calc_grads(layer4->grads_in);
        layer2->calc_grads(layer6->grads_in);
        layer1->calc_grads(layer2->grads_in);
        
        cout<<"************Fix weights**********";
        
        float diff_cost = cost_vec[cost_vec.size()-1] - cost_vec[cost_vec.size()-2];
        
        if(diff_cost < 1e-5){
            learning_rate /= 2.0;
        }

        layer1->fix_weights(learning_rate);
        layer2->fix_weights(learning_rate);
        // layer3->fix_weights(learning_rate);
        // layer4->fix_weights(learning_rate);
        // layer5->fix_weights(learning_rate);
        layer6->fix_weights(learning_rate);
        layer7->fix_weights(learning_rate);
        // layer8->fix_weights(learning_rate);
        // layer9->fix_weights(learning_rate);
        // layer10->fix_weights(learning_rate);
        layer11->fix_weights(learning_rate);
    }

    cout<<"**************cost*************\n";
    for(int i=0; i<cost_vec.size(); i++){
        cout<<cost_vec[i]<<endl;
    }

   /*************************************************************************hard coded code****************************************************************************/
                                                
                                                    /*not working great*/
//   activate((layer_t*)layer1, temp_in);
//   activate((layer_t*)layer10, ((layer_t*)layer1)->out);
//   activate((layer_t*)layer11, ((layer_t*)layer10)->out);

  
                                                    /*working great*/
//    activate((layer_t*)layer1, temp_in);
//    activate((layer_t*)layer10, layer1->out);
//    activate((layer_t*)layer11, layer10->out);

                                                    /*not working great*/
                                
    // activate(layers[0], temp_in);
    // activate(layers[1], layers[0]->out);
    // activate(layers[2], layers[1]->out);
    
                                                    /*working great*/
    // layer1->activate(temp_in);
    // layer10->activate(layer1->out);
    // layer11->activate(layer10->out);

/*************************************************************************loop code****************************************************************************/
    
	// layers.push_back( (layer_t*)layer1 );
	// layers.push_back( (layer_t*)layer2 );
	// layers.push_back( (layer_t*)layer3 );
	// layers.push_back( (layer_t*)layer4 );
    // layers.push_back( (layer_t*)layer5 );
    // layers.push_back( (layer_t*)layer6 );
    // layers.push_back( (layer_t*)layer7 );
    // layers.push_back( (layer_t*)layer8 );
    // layers.push_back( (layer_t*)layer9 );
    // layers.push_back( (layer_t*)layer10 );
    // layers.push_back( (layer_t*)layer11);
    // for ( int epoch = 0; epoch < 1; epoch++)
    // {
    //     for ( int i = 0; i < layers.size(); i++ )
    //     {
    //         if(layers[i]->type == layer_type::softmax){
    //             if(layers[i-1]->type == layer_type::scale){
    //                 // print_tensor(layers[i]->in);
    //                 // cout<<"flag3\n";
    //                 print_tensor(layers[i-1]->out);
    //             }

    //             // cout<<"**************\n";
    //             // print_tensor(layers[i]->in);
            
    //         }
    //         if ( i == 0 )
    //             activate( layers[i], temp_in);
    //         else
    //             activate( layers[i], layers[i - 1]->out);
    //     } 
    //     cout<<"\n\n\n\n\nloss************\n\n";
    //     cout<<cross_entropy(layer11->out, predict)(0, 0, 0, 0);

    //     int tm = layer11->out.size.m;
    //     int tx = layer11->out.size.x;
    //     tensor_t<float> softmax_grads(tm,tx,1,1);

    //     for(int i=0; i<tm; i++)
    //         for(int j=0; j<tx; j++)
    //             if(int(predict(i,j,0,0)) == 1)
    //                 softmax_grads(i,j,0,0) = (-1.0/layer11->out(i,j,0,0));
    
    //     for ( int i = layers.size() - 1; i >= 0; i-- )
    //         if ( i == layers.size() - 1 )
    //             calc_grads( layers[i], softmax_grads );
    //         else
    //             calc_grads( layers[i], layers[i + 1]->grads_in );

    //     cout<<"********* Fix weights ***********\n";

    //     for ( int i = 0; i < layers.size(); i++ )
    //         fix_weights( layers[i] );
    // }

    /*************************************************************************loop code****************************************************************************/
    return 0;
}