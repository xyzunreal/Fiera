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

    void train( string optimizer, tensor_t<float> input, tensor_t<float> output, int batch_size, int epochs=1, bool debug=false ){
       //TODO: Check layers are not empty

        int num_of_batches = input.size.m / batch_size ;

        for ( int epoch = 0; epoch < epochs; ++epoch){
            
            for(int batch_num = 0; batch_num<3; batch_num++)

            for ( int i = 0; i < layers.size(); i++ ){
		        if ( i == 0 )
		        	activate( layers[i], input );
                else
			        activate( layers[i], layers[i - 1]->out );
	            }

        }
    }
};