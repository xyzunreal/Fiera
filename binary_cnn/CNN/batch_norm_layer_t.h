#pragma once
#include<math.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct batch_norm_layer_t
{
	layer_type type = layer_type::batch_norm;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<float> in_hat;
	tensor_t<float> grads_in_hat,grads_xmu1,grads_xmu2,grads_x1;
	float epsilon;
	vector<gradient_t> grads_beta;
	vector<gradient_t> grads_gamma;
    std::vector<float> u_mean, gamma, beta, grads_sqrtvar,grads_var,grads_u_mean;
    std::vector<float> sigma, grads_sigma, grads_mean;
    bool adjust_variance;
	bool debug,clip_gradients_flag;

	batch_norm_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)
		:
		in(in_size.m, in_size.x, in_size.y, in_size.z ),
		grads_xmu1(in_size.m, in_size.x, in_size.y, in_size.z ),
		grads_xmu2(in_size.m, in_size.x, in_size.y, in_size.z ),
		grads_x1(in_size.m, in_size.x, in_size.y, in_size.z),
		out(in_size.m, in_size.x, in_size.y, in_size.z),
		in_hat(in_size.m, in_size.x, in_size.y, in_size.z),
		grads_in(in_size.m, in_size.x, in_size.y, in_size.z),
		grads_in_hat(in_size.m, in_size.x, in_size.y, in_size.z)
	{
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
		epsilon = 1e-3;
		gamma.resize(in.size.z, 1);
		beta.resize(in.size.z, 0);
        u_mean.resize(in.size.z);
		grads_sqrtvar.resize(in.size.z);
		grads_var.resize(in.size.z);
		grads_u_mean.resize(in.size.z);
        sigma.resize(in.size.z);
		grads_beta.resize(in.size.z);
		grads_gamma.resize(in.size.z);
        adjust_variance = false;
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
                        out(i,j,k,e) = gamma[e]*(in_hat(i,j,k,e)) + beta[e];
						// cout<<"out "<<out(i,j,k,e)<<endl;
                    }
                }
            }
        }
		
		// in_hat = in*gamma + beta

		if(debug){
			cout<<"\n**********output for batchnorm************\n";
			print_tensor(out);
		}
		
    }
	
	void activate()
	{
		cal_mean();
		cal_sigma();
		cal_in_hat();
	}
	
	
	
	void fix_weights(float learning_rate){
		
		for(int z=0; z<out.size.z; z++){
			update_weight(beta[z],grads_beta[z],1,false, learning_rate);
			update_gradient(grads_beta[z]);
			update_weight(gamma[z],grads_gamma[z],1,false, learning_rate);
			update_gradient(grads_gamma[z]);
		}
		if(debug)
		{
			// cout<<"\n***********updated beta********\n";
			// cout<<beta<<endl;
			// cout<<"\n***********update gamma*********\n";
			// cout<<gamma<<endl;
		}
	}

	void calc_grads( tensor_t<float>& grad_next_layer)
	{
		// fill(grads_beta.begin(), grads_beta.end(), 0);
		// fill(grads_gamma.begin(), grads_gamma.end(), 0);

		for(int z=0; z<out.size.z; z++){
			float sum = 0;
			for(int e=0; e<out.size.m; e++){
				for(int k=0; k<out.size.x; k++){
					for(int j=0; j<out.size.y; j++){
						sum += grad_next_layer(e,k,j,z);
					}
				}
			}
			grads_beta[z].grad = sum;	
		}

		for(int z=0; z<out.size.z; z++){
			float sum = 0;
			float sum1 = 0;
			for(int e=0; e<out.size.m; e++){
				for(int k=0; k<out.size.x; k++){
					for(int j=0; j<out.size.y; j++){
						sum += in_hat(e,k,j,z) * grad_next_layer(e,k,j,z);
						grads_in_hat(e,k,j,z) = grad_next_layer(e,k,j,z)*gamma[z]; 
						sum1 += grads_in_hat(e,k,j,z)*(in(e,k,j,z) - u_mean[z]); 
						grads_xmu1(e,k,j,z) = grads_in_hat(e,k,j,z)/sqrt(epsilon+sigma[z]); 
					}
				}
			}	
			grads_gamma[z].grad = sum;
			// grads_ivar[z] = sum1;
			grads_sqrtvar[z] = -1*sum1/(epsilon+sigma[z]);
			grads_var[z] = 0.5 * (grads_sqrtvar[z]/sqrt(epsilon+sigma[z]));
		}
		
		for(int z = 0; z<out.size.z; z++){
			float sum = 0;
			for(int e=0; e<out.size.m; e++){
				for(int k=0; k<out.size.x; k++){
					for(int j=0; j<out.size.y; j++){
						// grads_xmu2(e,k,j,z) = (1.0/out.size.m)*grads_var[z]*2*(in(e,k,j,z) - u_mean[z]);
						grads_xmu2(e,k,j,z) = (grads_var[z]*2*(in(e,k,j,z) - u_mean[z]))/(out.size.m*out.size.x*out.size.y);
						grads_x1(e,k,j,z) = grads_xmu1(e,k,j,z) + grads_xmu2(e,k,j,z);
						sum += grads_x1(e,k,j,z);
					}
				}
			}
			grads_u_mean[z] = -sum; 
			// cout<<"sum "<<sum<<' ';
		}

		// cout<<"grads_xmul\n";
		// print_tensor(grads_x1);
		
		for(int z = 0; z<out.size.z; z++){
			for(int e=0; e<out.size.m; e++){
				for(int k=0; k<out.size.x; k++){
					for(int j=0; j<out.size.y; j++){
						// grads_in(e,k,j,z) = grads_x1(e,k,j,z) + (1.0/out.size.m)*grads_u_mean[z];
						grads_in(e,k,j,z) = (grads_x1(e,k,j,z) + grads_u_mean[z])/(out.size.m*out.size.x*out.size.y);
					}
				}
			}
		}

		// for(int i=0; i<out.size.z; i++){

		// 	float temp_sigma = 0;
		// 	float temp_mean = 0;
		// 	float temp_mean1 = 0;
		// 	float temp_mean2 = 0;

		// 	for(int e=0; e<out.size.m; e++){
		// 		for(int k=0; k<out.size.x; k++){
		// 			for(int j=0; j<out.size.y; j++){
		// 				grads_beta.grad += grad_next_layer(e,k,j,i); 
		// 				clip_gradients(clip_gradients_flag, grads_beta.grad);

		// 				grads_gamma.grad += grad_next_layer(e,k,j,i)*in_hat(e,k,j,i);
		// 				clip_gradients(clip_gradients_flag, grads_gamma.grad);
						
		// 				grads_in_hat(e,k,j,i) = grad_next_layer(e,k,j,i)*gamma;
		// 				temp_sigma += (grads_in_hat(e, k, j, i)*(in(e,k,j,i)-u_mean[i])*(-1.0/2.0)*(pow((sigma[i]+epsilon),-3/2)));
		// 				temp_mean1 += (grads_in_hat(e, k, j, i)*(in(e,k,j,i)-u_mean[i])*(-1.0/sqrt(sigma[i] + epsilon)));
		// 				temp_mean2 += (-2 * (in(e, k, j, i) - u_mean[i]));
		// 			}
		// 		}
		// 	}
		// 	temp_mean2 *= (grads_sigma[i])/(out.size.m);
		// 	grads_sigma[i] = temp_sigma;
		// 	grads_mean[i] = temp_mean1 + temp_mean2;

		// }

		// for(int i=0; i<out.size.z; i++){
		// 	for(int e=0; e<out.size.m; e++){
		// 		for(int k=0; k<out.size.x; k++){
		// 			for(int j=0; j<out.size.y; j++){
		// 				grads_in(e,j,k,i) = grads_in_hat(e, j, k, i)* (1.0/sqrt(sigma[i]+epsilon)) + grads_sigma[i] * (2.0 * (in(e, j, k, i) - u_mean[i])/ out.size.m ) + grads_mean[i]/out.size.m;
		// 				clip_gradients(clip_gradients_flag, grads_in(e,j,k,i));
		// 			}
		// 		}
		// 	}

		// }

		if(debug)
		{
			cout<<"\n*********grads_in for batch_norm************\n";
			print_tensor(grads_in);
		}	
	}
};
#pragma pack(pop)
