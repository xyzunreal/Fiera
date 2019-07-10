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
    tensor_t<float> temp_in(2, 3, 3, 2), filters(2, 2, 2, 2), expected_output(2, 2, 2, 2);
      std::vector<std::vector<std::vector<std::vector<float> > > > vect=
         {{{{-2.2578,  0.4043,  0.5722},
          { 0.3078, -0.1259, -0.9578},
          { 1.7518,  0.9796, -0.3985}},

         {{-0.1732,  0.7569,  0.9862},
          {-0.8253,  0.1633,  0.5013},
          { 1.4206, -0.5368,  0.8289}}},


        {{{ 1.0571, -1.1047, -0.5860},
          {-0.4896,  0.6398,  0.6094},
          {-0.3838, -0.2675,  0.6794}},

         {{-0.3947, -0.4311,  0.6133},
          { 0.2830,  1.1855, -1.2980},
          {-0.7873,  1.7422,  0.5880}}}};

    temp_in.from_vector(vect);
    print_tensor(temp_in);
	
   vect = {{{{ 0.1701, -0.1747},
          {-0.1887,  0.3051}},

         {{ 0.3235,  0.0407},
          {-0.0612, -0.0456}}},


        {{{ 0.1675, -0.3301},
          {-0.2889,  0.2824}},

         {{ 0.3490, -0.0210},
          {-0.2794,  0.0097}}}};

              
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
    // layer->in = temp_in;
    layer->filters = filters;
    cout<<"\nfilters\n";
    print_tensor(layer->filters);
    
    tensor_t<float> out;

    out = layer->activate(temp_in);
    if (out == expected_output) cout << "Convlayer forward working correctly";
    else{
        cout << "Expected output is\n";
        print_tensor(expected_output);
        cout << "\nActual output is\n";
        print_tensor(out);
    }
}