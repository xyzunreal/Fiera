#pragma once
#include "layer_t.h"

struct scale_layer_t
{
    layer_type type = layer_type::scale;
    tensor_t<float> grads_in;
    tensor_t<float> in;
    tensor_t<float> out;
    gradient_t grads_scale;
    bool debug,clip_gradients_flag;

    float s_param;     // scaler learnable parameter
    tensor_t<gradient_t> gradients;

    scale_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)        // EXPECTS 1D INPUT
        :
        in(in_size.m, in_size.x, 1, 1),
        out(in_size.m, in_size.x, 1, 1),
        grads_in(in_size.m, in_size.x, 1, 1),
        gradients(in_size.m, in_size.x, 1, 1)
        
    {
        s_param = 0.001; 
        this->debug = debug_flag;    
        this->clip_gradients_flag = clip_gradients_flag;
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
            }

        if(debug)
        {
            cout<<"*****output for scale***********\n";
            print_tensor(out);
        }
    }

    void fix_weights(float learning_rate)
    {
        // grads_scale contains sum of gradients of s_param for all examples. 
		grads_scale.grad /= out.size.m;
		s_param = update_weight(s_param,grads_scale,1,false, learning_rate);
		update_gradient(grads_scale);
       
        if(debug)
        {
            cout<<"*******updated s_param*****\n";
		    cout<<s_param<<endl;
        }
    }

    void calc_grads(tensor_t<float>& grad_next_layer)
    {
        grads_scale.grad = 0;
        for(int i=0; i<out.size.m; i++)
            for(int j=0; j<out.size.x; j++){
                grads_scale.grad += grad_next_layer(i,j,0,0)*in(i,j,0,0); 
                grads_in(i,j,0,0) = grad_next_layer(i,j,0,0)*s_param;
                clip_gradients(clip_gradients_flag, grads_scale.grad);
                clip_gradients(clip_gradients_flag, grads_in(i,j,0,0));
            }
        if(debug)
        {
            cout<<"***********grads_in for scale********\n";
            print_tensor(grads_in);
            cout<<"***********gradient for s_param*******\n";
            cout<<grads_scale.grad<<endl;
        }

    }

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "scale" },
			{ "in_size", {in.size.x, in.size.y, in.size.z, in.size.m} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

};