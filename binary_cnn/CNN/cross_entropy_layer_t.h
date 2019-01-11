/*
To calculate cross entropy loss
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
float crossentropy_layer_t(tensor_t<float>& predicted ,tensor_t<float>& actual){
        int index;
        tensor_t<float> temp;
        for(int e=0; e < predicted.size.m; e++){
            for ( int i = 0; i < predicted.size.x; i++ ){
                if( int(actual(e,i, 0, 0)) == 1){
                    index=i;
                    break;
                }	
            }
            temp.push_back(-log(predicted(e,index,0,0)));
        }
        return temp; 
    }
