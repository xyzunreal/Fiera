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
    tensor_t<float> temp_in(2, 3, 3, 2), filters(2, 2, 2, 2), expected_output(2, 2, 2, 2), grad_next_layer(2, 2, 2, 2), diff_weights(2,2,2,2);
    std::vector<std::vector<std::vector<std::vector<float> > > > vect=

      {{{{-0.1782, -0.2595, -0.0145},
          {-0.3839, -2.9662, -1.0606},
          {-0.3090,  0.9343,  1.6243}},

         {{ 0.0016, -0.4375, -2.1085},
          { 1.1450, -0.3822, -0.3553},
          { 0.7542,  0.1332,  0.1825}}},


        {{{-0.5146,  0.8005, -0.1259},
          {-0.9578,  1.7518,  0.9796},
          { 0.4105,  1.7675, -0.0832}},

         {{ 0.5087, -0.8253,  0.1633},
          { 0.5013,  1.4206,  1.1542},
          {-1.5366, -0.5577, -0.4383}}}};

    temp_in.from_vector(vect);
    cout<<"*********image*****\n\n";
    print_tensor(temp_in);

    vect = {{{{ 0.0000,  0.0000},
          { 0.1686, -0.0938}},

         {{ 0.0000,  0.0000},
          { 0.0000,  0.0220}}},


        {{{ 0.0000, -0.0775},
          { 0.0000,  0.0000}},

         {{-0.0039,  0.0734},
          {-0.0856, -0.0569}}}};
      
      grad_next_layer.from_vector(vect);
      cout<<"**********grad_next_layer**********\n";
      print_tensor(grad_next_layer);

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


    conv_layer_t * layer = new conv_layer_t( 1, 2, 2, temp_in.size, false);
    layer->in = temp_in;
    layer->filters = filters;

    cout<<"**********old filters ************\n";
    print_tensor(filters);
    
    layer->activate();
    layer->calc_grads(grad_next_layer);


     
    layer->fix_weights(1);
    cout<<"**********new filters************\n";
    print_tensor(layer->filters);

    diff_weights = layer->filters - filters;
    cout<<"*********diff weights";
    print_tensor(diff_weights);
}
