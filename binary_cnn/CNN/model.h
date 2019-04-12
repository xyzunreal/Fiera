#include <cassert>
#include <cstdint>
#include <string.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "cnn.h"

using namespace std;

class Model{

    vector<layer_t* > layers;

    Model(vector<layer_t* > layers){
        this->layers = layers;
    }

    void train( string optimizer, tensor_t<float> input, tensor_t<float> output, int batch_size, int epochs=1, float lr = 0.02, bool debug=false ){
       //TODO: Check layers are not empty

        int num_of_batches = input.size.m / batch_size ;
        float loss;

        for ( int epoch = 0; epoch < epochs; ++epoch){
            for(int batch_num = 0; batch_num<num_of_batches; batch_num++)
            {
                tensor_t<float> input_batch = input.get_batch(batch_size, batch_num);
                tensor_t<float> output_batch = output.get_batch(batch_size, batch_num);

                // Forward propogate
                for ( int i = 0; i < layers.size(); i++ )
                {
                    if ( i == 0 )
                        activate( layers[i], input_batch );
                    else
                        activate( layers[i], layers[i - 1]->out );
                }

                // Calculate Loss
                loss = cross_entropy(layers[layers.size()-1]->out, output_batch, debug);

                // Backpropogation
                for ( int i = layers.size() - 1; i >= 0; i-- )
                {
                    if ( i == layers.size() - 1 )
                        calc_grads( layers[i], output_batch);
                    else
                        calc_grads( layers[i], layers[i + 1]->grads_in );
                }

                // Update weights
                for ( int i = 0; i < layers.size(); i++ )
                    fix_weights( layers[i], lr);
            }
            cout<<"Loss after epoch "<<epoch<<" : "<< loss <<endl;
        }

        void save_model(string fileName){
            ofstream file(fileName);
            json model;
            model["config"] = {
                {}
            }
            for ( int i = 0; i < layers.size(); i++ ) 
                if ( i == layers.size() - 1 ){
                    save_layer( layer[i], model );
                }
        }
    }




};