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
    tensor_t<float> temp_in(2, 3, 3, 2), filters(2, 2, 2, 2), expected_output(2, 2, 2, 2), grad_next_layer(2, 2, 2, 2);
    std::vector<std::vector<std::vector<std::vector<float> > > > vect=

       {{{{ 1.4065e-01,  1.0983e+00, -8.9477e-02},
          { 4.2327e-01,  7.3052e-01,  7.5187e-01},
          { 7.6115e-02, -4.1352e-01,  6.9499e-01}},

         {{ 8.9989e-02, -1.6552e-01,  5.0196e-02},
          {-1.6791e-01, -1.4499e-01, -1.6246e-01},
          { 6.6821e-05,  1.3456e-01, -8.3133e-02}}},


        {{{-1.0812e+00,  4.4041e-01,  7.8283e-01},
          { 5.9163e-02,  1.7522e-01, -8.8939e-02},
          { 2.4275e-01,  5.9299e-02,  4.0713e-01}},

         {{ 2.6039e-01, -1.0425e-01, -7.7286e-02},
          { 3.9525e-04,  7.0601e-03, -9.3474e-02},
          { 7.1038e-02,  1.3932e-02, -1.1881e-01}}}};

    temp_in.from_vector(vect);
    cout<<"*********image*****\n\n";
    print_tensor(temp_in);

    vect = {{{{-0.1074,  0.0791},
          {-0.0208,  0.0348}},

         {{ 0.0871,  0.0792},
          {-0.0607, -0.1115}}},


        {{{ 0.0173, -0.0171},
          {-0.0074, -0.1149}},

         {{-0.0512, -0.0225},
          { 0.0215,  0.0688}}}};
      
      grad_next_layer.from_vector(vect);
      cout<<"**********grad_next_layer**********\n";
      print_tensor(grad_next_layer);

    vect = {{{{ 0.1675, -0.3301},
          {-0.2889,  0.2824}},

         {{ 0.3490, -0.0210},
          {-0.2794,  0.0097}}},


        {{{-0.1645, -0.0007},
          { 0.1731,  0.1565}},

         {{-0.0415,  0.0389},
          { 0.0962, -0.2771}}}};
              
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


    conv_layer_bin_t * layer = new conv_layer_bin_t( 1, 2, 2, temp_in.size, true);
    layer->in = temp_in;
    layer->filters = filters;

    cout<<"**********filters weights************\n";
    print_tensor(layer->filters);
    
    // tensor_t<float> out;

    layer->activate(temp_in, true);


    tensor_t<float> grads_in;
    
    layer->calc_grads(grad_next_layer);
    cout<<"flag3\n";


    // cout<<"**********grads calculated************\n";
    // print_tensor(grads_in);
    


    // if (layer->out == expected_output) cout << "Convlayer forward working correctly";
    // else{
        // cout << "Expected output is\n";
        // print_tensor(expected_output);
        // cout << "\nActual output is\n";
        // print_tensor(layer->out);
    // }
}
