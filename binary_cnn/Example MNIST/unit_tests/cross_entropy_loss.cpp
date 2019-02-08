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
    tensor_t<float> temp_in(2, 4, 1, 1),  target(2,4,1,1), loss(2,1,1,1), expected_output(2, 1, 1, 1);
    float expected_loss = 1.3562, actual_loss;

    std::vector<std::vector<std::vector<std::vector<float> > > > vect=
       {{{{ 0.0308,  0.1925,  0.1382,  0.0727}}},
        {{{-0.0132,  0.0684, -0.1085, -0.0739}}}};

    temp_in.from_vector(vect);
    softmax_layer_t * softmax = new softmax_layer_t({2,4,1,1}, true);
    softmax->in = temp_in;
    softmax->activate();
    // print_tensor(softmax->out);

    // Defining target values
    target(0, 3, 0, 0) = 1;     // For 1st example, target is 3. i.e 4th class
    target(1, 1, 0, 0) = 1;     // For 2nd example, target is 1. i.e 2th class
     
    // loss = cross_entropy(softmax->out, target);
    // cross_entropy(softmax->out, target)(1,0,0,0)<<endl;
    // cout<<"loss tensor: \n";
    // print_tensor(loss);

    // cout<<loss(0,0,0,0)<<endl;
    cout<<cross_entropy(softmax->out, target)(0,0,0,0)<<endl;
    actual_loss = cross_entropy(softmax->out, target)(0,0,0,0) + cross_entropy(softmax->out, target)(1,0,0,0);
    cout << actual_loss/2 <<"\n\n"<< expected_loss;

}