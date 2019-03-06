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
    tensor_t<double> temp_in(2, 2, 2, 2), weights(2 * 2 * 2, 4, 1, 1), expected_output(2, 4, 1, 1), grad_next_layer(2,4,1,1), diff_weights(2*2*2,4,1,1);
    std::vector<std::vector<std::vector<std::vector<double> > > > vect=

        {{{{0.0000, 0.0000},
          {0.3510, 0.5182}},

         {{0.0000, 0.0000},
          {0.0000, 0.4201}}},


        {{{0.0000, 0.3114},
          {0.0000, 0.0000}},

         {{0.0045, 0.2879},
          {0.4376, 0.1286}}}};

    temp_in.from_vector(vect);

    vect = 
        {{{{ 0.1558,  0.0148,  0.0896,  0.170}}},
        {{{-0.3019,  0.0659,  0.0488, -0.1747}}},
        {{{ 0.3323,  0.2685,  0.1723, -0.1887}}},
        {{{-0.2773,  0.0909,  0.3247,  0.3051}}},
        {{{ 0.2707,  0.1876, -0.0787,  0.3235}}},
        {{{-0.0614, -0.2735, -0.1970,  0.0407}}},
        {{{ 0.1819,  0.2517, -0.0890, -0.0612}}},
        {{{ 0.1378,  0.1217, -0.2155, -0.0456}}}};

    weights.from_vector(vect);

    // vect =
    //   {{{{ 0.1630, -0.2220, -0.1793,  0.5250}}},
    //       {{{ 0.5219, -0.1114, -0.3139,  0.3302}}}};

    // expected_output.from_vector(vect);

    vect = 
      {{{{ 0.1154,  0.1357,  0.1285, -0.3796}}},
        {{{0.1271, -0.3621,  0.1155,  0.1196}}}};

    grad_next_layer.from_vector(vect);
    
    fc_layer_t * layer = new fc_layer_t( {2, 2, 2, 2}, 4, true);
    layer->in = temp_in;
    
    cout<<"**********old filters ************\n";
    print_tensor(weights);
    layer->weights = weights;
    layer->activate();
    layer->calc_grads(grad_next_layer);

    layer->fix_weights(1);
    cout<<"**********new filters************\n";
    print_tensor(layer->weights);

    diff_weights = layer->weights - weights;
    cout<<"*********diff weights";
    print_tensor(diff_weights);
    }