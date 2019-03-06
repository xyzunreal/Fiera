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
    tensor_t<double> temp_in(2, 2, 2, 2), weights(2 * 2 * 2, 4, 1, 1), expected_output(2, 4, 1, 1);
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

    vect =
       {{{{ 0.0308,  0.1925,  0.1382,  0.0727}}},
        {{{-0.0132,  0.0684, -0.1085, -0.0739}}}};

    expected_output.from_vector(vect);

    fc_layer_t * layer = new fc_layer_t( {2, 2, 2, 2}, 4, false);
    layer->in = temp_in;
    layer->weights = weights;
    layer->activate();
    cout << "\nExpected output is\n";
    print_tensor(expected_output);
    cout << "\nActual output is\n";
    print_tensor(layer->out);
    }