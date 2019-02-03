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
using namespace std;
tensor_t<float> cross_entropy(tensor_t<float>& predicted ,tensor_t<float>& actual, bool debug=false){
        int index;
        tensor_t<float> temp(predicted.size.m, 1, 1, 1);
        if(debug)
        {
            cout<<"***********predicted*********";
            print_tensor(predicted);

            cout<<"***********actual***********";
            print_tensor(actual);
        }
        for(int e=0; e < predicted.size.m; e++){
            for ( int i = 0; i < predicted.size.x; i++ ){
                if( int(actual(e,i, 0, 0)) == 1){
                    index=i;
                    break;
                }	
            }
            temp(e,0,0,0) = (-log(predicted(e,index,0,0)));
        }
    
        return temp; 
    }
