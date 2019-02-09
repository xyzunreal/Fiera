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
    tensor_t<float> in(2, 2, 2, 2), actual_output(2, 2, 2, 2), expected_output(2, 2, 2, 2);
    float gamma, beta, epsilon;

    std::vector<std::vector<std::vector<std::vector<float> > > > vect=
        {{{{ 0.4083,  0.4021},
          {-0.0062, -0.0193}},

         {{-0.0176, -0.0468},
          {-0.0080, -0.0182}}},


        {{{-0.0064, -0.0038},
          { 0.0441, -0.0048}},

         {{-0.0107,  0.5100},
          { 0.0531,  0.8615}}}};
    in.from_vector(vect);

    vect = 
        {{{{ 1.2540,  1.2287},
          {-0.4414, -0.4951}},

         {{-0.0426, -0.0493},
          {-0.0403, -0.0427}}},


        {{{-0.4426, -0.4318},
          {-0.2358, -0.4360}},

         {{-0.0410,  0.0801},
          {-0.0261,  0.1619}}}};
    expected_output.from_vector(vect);

    beta = 0.7204;
    gamma = 0.0731;
    epsilon = 1e-5;

    batch_norm_layer_t * layer = new batch_norm_layer_t({2, 2, 2, 2});
    layer->in = in;
    layer->gamma = gamma;
    layer->beta = beta;
    layer->epsilon = epsilon;
    layer->adjust_variance = false;
    layer->activate();

    if (layer->out == expected_output) 
        cout << "Batch Norm working correctly";
    else 
        cout << "Batch Norm not working correctly";

        cout << "\n\n Input image";
        print_tensor(layer->in);
        cout << "\n\n Expected output";
        print_tensor(expected_output);
        cout << "\n Actual output";
        print_tensor(layer->out);
        cout << "\n Mean \n";
        for (int i=0;i<layer->u_mean.size();i++)
            cout << layer->u_mean[i] << "  ";
        cout << "\n Sigma \n";
        for (int i=0;i<layer->sigma.size();i++)
            cout<< layer->sigma[i] << "  ";
        
        cout << "\n Gamma\t" << gamma;
        cout << "\n beta\t" << beta;
}