#pragma once
#include "layer_t.h"
#pragma pack(push, 1)

struct scale_layer_t
{
    layer_type type = layer_type::scale;
    tensor_t<float> grads_in;
    gradient_t grads_scale;
    tensor_t<float> in;
    tensor_t<float> out;

    float s_param;     // `s_param` REPRESENT SCALE VALUE WHICH IS SINGLE LEARNABLE PARAMETER.
    tensor_t<gradient_t> gradients;

    scale_layer_t(tdsize in_size)        // EXPECTS 1D INPUT
        :
        in(in_size.m, in_size.x, 1, 1),
        out(in_size.m, in_size.x, 1, 1),
        grads_in(in_size.m, in_size.x, 1, 1),
        gradients(in_size.m, in_size.x, 1, 1)
        
    {
        s_param = 0.001;    
    }

    void activate( tensor_t<float> & in )
    {
        this->in = in;
        activate();
    }

    void activate()
    {
		for (int tm = 0; tm < in.size.m; tm++)
			for (int i = 0; i < in.size.x; i++){
				out(tm,i, 0, 0) = s_param * in(tm, i, 0, 0);
                // cout<<s_param<<' '<<in(tm, i, 0, 0)<<' '<<s_param * in(tm, i, 0, 0)<<endl;
            }
        cout<<"*****output for scale***********\n";
        print_tensor(out);
        // cout<<"************flag********\n";
	    // cout<<out.size.m<<' '<<out.size.x<<' '<<out.size.y<<" "<<out.size.z<<endl;
    }

    void fix_weights()
    {
        // grads_scale contains sum of gradients of s_param for all examples. 
		grads_scale.grad /= out.size.m;
		update_weight(s_param,grads_scale);
		update_gradient(grads_scale);
    }

    void calc_grads(tensor_t<float>& grad_next_layer)
    {
        grads_scale.grad = 0;
        for(int i=0; i<out.size.m; i++){
            for(int j=0; j<out.size.x; j++){
                grads_scale.grad += grad_next_layer(i,j,0,0)*in(i,j,0,0); 
                grads_in(i,j,0,0) = grad_next_layer(i,j,0,0)*s_param;
            }
        }
    }

};
#pragma pack(pop)