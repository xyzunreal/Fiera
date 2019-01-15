#pragma once
#include<math.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct batch_norm_layer_t
{
	layer_type type = layer_type::batch_norm;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<float> in_hat;
	tensor_t<float> grads_in, grads_in_hat;
	float epsilon;
    float gamma;
    float beta;
	gradient_t grads_beta;
	gradient_t grads_gamma;
    std::vector<float> u_mean;
    std::vector<float> sigma, grads_sigma, grads_mean;
    bool adjust_variance;
	
	batch_norm_layer_t(tdsize in_size )
		:
		in(in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m, in_size.x, in_size.y, in_size.z),
		in_hat(in_size.m, in_size.x, in_size.y, in_size.z),
		grads_in(in_size.m, in_size.x, in_size.y, in_size.z),
		grads_in_hat(in_size.m, in_size.x, in_size.y, in_size.z)
	{
		epsilon = 1e-3;
        gamma = 1;
        beta = 0;
        u_mean.resize(in.size.z);
        sigma.resize(in.size.z);
        adjust_variance = true;
		grads_sigma.resize(in.size.z);
		grads_mean.resize(in.size.z);
	}

	

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}
	
	void cal_mean(){
        for(int i=0; i<in.size.z; i++){
            float sum = 0;
            for(int j=0; j<in.size.m; j++){
                for(int k = 0; k<in.size.x; k++){
                    for(int m = 0; m<in.size.y; m++){
                        sum += in(j,k,m,i);
                    }
                }
            }
            sum /= (in.size.x*in.size.m*in.size.y);
            u_mean[i] = sum;
        }        
    }
	
	void cal_sigma(){
         for(int i=0; i<in.size.z; i++){
            float sum = 0;
            for(int j=0; j<in.size.m; j++){
                for(int k = 0; k<in.size.x; k++){
                    for(int m = 0; m<in.size.y; m++){
                        sum += ((in(j,k,m,i) - u_mean[i])*(in(j,k,m,i) - u_mean[i]));
                    }
                }
            }
            sum /= (in.size.x*in.size.m*in.size.y);
            sigma[i] = sum;
        }

        if(in.size.m > 1 and adjust_variance){
            for(int i=0; i<sigma.size(); i++){
                sigma[i] *= ((in.size.m)/(in.size.m-1));
            }
        }
    }
	
	void cal_in_hat(){
        for(int e=0; e<in.size.z; e++){
            for(int i=0; i<in.size.m; i++){
                for(int j=0; j<in.size.x; j++){
                    for(int k=0; k<in.size.y; k++){
						in_hat(i,j,k,e) = (in(i,j,k,e)-u_mean[e])/sqrt(sigma[e]+epsilon);
                        out(i,j,k,e) = gamma*(in_hat(i,j,k,e)) + beta;
                    }
                }
            }
        }
    }
	void activate()
	{
		cal_mean();
		cal_sigma();
		cal_in_hat();
	}
	
	
	
	void fix_weights(){
		grads_beta.grad /= out.size.m;
		grads_gamma.grad /= out.size.m;
		update_weight(beta,grads_beta);
		update_gradient(grads_beta);
		update_weight(gamma,grads_gamma);
		update_gradient(grads_gamma);
	}

	void calc_grads( tensor_t<float>& grad_next_layer)
	{
		grads_beta.grad = 0;
		grads_gamma.grad = 0;

		for(int i=0; i<out.size.z; i++){

			float temp_sigma = 0;
			float temp_mean = 0;
			float temp_mean1 = 0;
			float temp_mean2 = 0;

			for(int e=0; e<out.size.m; e++){
				for(int k=0; k<out.size.x; k++){
					for(int j=0; j<out.size.y; j++){
						grads_beta.grad += grad_next_layer(e,k,j,i); 
						grads_gamma.grad += grad_next_layer(e,k,j,i)*in_hat(e,k,j,i);
						grads_in_hat(e,k,j,i) = grad_next_layer(e,k,j,i)*gamma;
						temp_sigma += (grads_in_hat(e, k, j, i)*(in(e,k,j,i)-u_mean[i])*(-1.0/2.0)*(pow((sigma[i]+epsilon),-3/2)));
						temp_mean1 += (grads_in_hat(e, k, j, i)*(in(e,k,j,i)-u_mean[i])*(-1.0/sqrt(sigma[i] + epsilon)));
						temp_mean2 += (-2 * (in(e, k, j, i) - u_mean[i]));
					}
				}
			}
			temp_mean2 *= (grads_sigma[i])/(out.size.m);
			grads_sigma[i] = temp_sigma;
			grads_mean[i] = temp_mean1 + temp_mean2;

		}

		for(int i=0; i<out.size.z; i++){
			for(int e=0; e<out.size.m; e++){
				for(int k=0; k<out.size.x; k++){
					for(int j=0; j<out.size.y; j++){
						grads_in(e,j,k,i) = grads_in_hat(e, j, k, i)* (1.0/sqrt(sigma[i]+epsilon)) + grads_sigma[i] * (2.0 * (in(e, j, k, i) - u_mean[i])/ out.size.m ) + grads_mean[i]/out.size.m;
					}
				}
			}

		}	
	}
};
#pragma pack(pop)
