#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../../CNN/cnn.h"

using namespace std;

int main()
{
    tensor_t<double> filter_grads(2, 2, 2, 2), filters(2, 2, 2, 2), expected_output(2, 2, 2, 2), grad_next_layer(2, 2, 2, 2), diff_weights(2,2,2,2);
    std::vector<std::vector<std::vector<std::vector<double> > > > vect=

      {{{{ 0.1515, -0.3932},
          {-0.2842, -0.0689}},

         {{ 0.2915, -0.0537},
          { 0.0069, -0.0797}}},


        {{{-0.0204, -0.2494},
          { 0.0300, -0.0272}},

         {{-0.1886, -0.1701},
          { 0.2679,  0.1591}}}};

    filter_grads.from_vector(vect);
    cout<<"*********conv dw*****\n\n";
    print_tensor(filter_grads);

   
    vect = {{{{ 0.0247, -0.2130},
              { 0.1126,  0.1109}},

             {{-0.1890, -0.0530},
              {-0.2071,  0.0917}}},

            {{{-0.0952,  0.2484},
              { 0.2510,  0.0360}},

             {{-0.1507, -0.2077},
              {-0.0388, -0.0995}}}};
              
    filters.from_vector(vect);

    vect = {{{{ 0.1503,  0.0721},
          {-0.4510,  0.3723}},

         {{ 0.3587, -0.3715},
          {-0.1152, -0.3853}}},


        {{{ 0.4237,  0.0047},
          { 0.4470, -0.0187}},

         {{ 0.2511,  0.2050},
          { 0.2774,  0.2570}}}};
    
    expected_output.from_vector(vect);


    conv_layer_t * layer = new conv_layer_t( 1, 2, 2, {2,3,3,2}, false);
    layer->filters = filters;
    
    for(int i=0; i<filter_grads.size.m; i++){
      for(int j=0; j<filter_grads.size.x; j++){
        for(int k=0; k<filter_grads.size.y; k++){
          for(int l=0; l<filter_grads.size.z; l++){
            layer->filter_grads(i,j,k,l).grad = filter_grads(i,j,k,l);      
          }
        }
      }
    }
    cout<<"**********old filters ************\n";
    print_tensor(filters);
    
    layer->fix_weights(0.01);

    cout<<"**********new filters************\n";
    print_tensor(layer->filters);

    diff_weights = layer->filters - filters;
    cout<<"*********diff weights";
    print_tensor(diff_weights);
}
