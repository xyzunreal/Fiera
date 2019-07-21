/*! 
* Batch Norm Layer
*/

//TODO: Adding debug flags to ifdef

#pragma once
#include <math.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct batch_norm_layer_t
{
	layer_type type = layer_type::batch_norm;
	tensor_t<float> in;
	tensor_t<float> in_hat;
	tensor_t<float> out;
	tdsize in_size, out_size;
	float epsilon;
	vector<float> gamma, beta, u_mean, sigma;
	vector<gradient_t> grads_beta, grads_gamma;
    bool adjust_variance;
	bool debug, clip_gradients_flag;

	batch_norm_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)
	/**
	* 
	* Parameters
	* ----------
	* in_size : tdsize
	* 		size of input
	*
	* clip_gradients_flag : bool
	* 		Whether gradients have to be clipped or not
	* 
	* debug_flag : bool
	* 		Whether to print variables for debugging purpose
	*
	**/
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
		epsilon = 1e-5;
		gamma.resize(in_size.z, 1);
		beta.resize(in_size.z, 0);
		u_mean.resize(in_size.z);
		sigma.resize(in_size.z);
        adjust_variance = false;
	}

	tensor_t<float> activate(tensor_t<float> & in, bool train = true){
		if(train) this->in = in;

		cal_mean(in);
		cal_sigma(in);
		return cal_in_hat(in,train);
	}

	void cal_mean(tensor_t<float> in){
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
	
	void cal_sigma(tensor_t<float> in){
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
	
	tensor_t<float> cal_in_hat(tensor_t<float> in, bool train){
		
		tensor_t<float> out(in.size.m, in.size.x, in.size.y, in.size.z);
		tensor_t<float> in_hat(in.size);

        for(int e=0; e<in.size.z; e++){
            for(int i=0; i<in.size.m; i++){
                for(int j=0; j<in.size.x; j++){
                    for(int k=0; k<in.size.y; k++){
						in_hat(i,j,k,e) = (in(i,j,k,e)-u_mean[e])/sqrt(sigma[e]+epsilon);
                        out(i,j,k,e) = gamma[e]*(in_hat(i,j,k,e)) + beta[e];
                    }
                }
            }
        }

		if(train) {
			this->out = out;
			this->in_hat = in_hat;
		}

		return out;	
    }
	
	
	void fix_weights(float learning_rate){
		
		for(int z=0; z<in_size.z; z++){
			beta[z] = update_weight(beta[z],grads_beta[z],1,false, learning_rate);
			update_gradient(grads_beta[z]);
			gamma[z] = update_weight(gamma[z],grads_gamma[z],1,false, learning_rate);
			update_gradient(grads_gamma[z]);
		}
	}

	tensor_t<float> calc_grads( tensor_t<float>& grad_next_layer)
	{
		assert(in.size > 0); 

		tensor_t <float> grads_xmu1(grad_next_layer.size.m, in_size.x,in_size.y, in_size.z),
      		grads_xmu2(grad_next_layer.size.m, in_size.x, in_size.y, in_size.z),
      		grads_x1(grad_next_layer.size.m, in_size.x, in_size.y, in_size.z),
      		grads_in(grad_next_layer.size.m, in_size.x, in_size.y, in_size.z),
      		grads_in_hat(grad_next_layer.size.m, in_size.x, in_size.y, in_size.z);

		vector<float> grads_sqrtvar,grads_var,grads_u_mean;
    	vector<float> grads_sigma, grads_mean;

		grads_gamma.resize(in_size.z);
		grads_beta.resize(in_size.z);

		grads_sqrtvar.resize(in_size.z);
		grads_var.resize(in_size.z);
		grads_u_mean.resize(in_size.z);
		grads_sigma.resize(in_size.z);
		grads_mean.resize(in_size.z);

		for(int z=0; z<in_size.z; z++){
			float sum = 0;
			for(int e=0; e<grad_next_layer.size.m; e++){
				for(int k=0; k<in_size.x; k++){
					for(int j=0; j<in_size.y; j++){
						sum += grad_next_layer(e,k,j,z);
					}
				}
			}
			grads_beta[z].grad = sum;	
		}

		for(int z=0; z<in_size.z; z++){
			float sum = 0;
			float sum1 = 0;
			for(int e=0; e<grad_next_layer.size.m; e++){
				for(int k=0; k<in_size.x; k++){
					for(int j=0; j<in_size.y; j++){
						sum += in_hat(e,k,j,z) * grad_next_layer(e,k,j,z);
						grads_in_hat(e,k,j,z) = grad_next_layer(e,k,j,z)*gamma[z]; 
						sum1 += grads_in_hat(e,k,j,z)*(in(e,k,j,z) - u_mean[z]); 
						grads_xmu1(e,k,j,z) = grad_next_layer(e,k,j,z) * (gamma[z]) /sqrt(epsilon+sigma[z]); 
					}
				}
			}	
			grads_gamma[z].grad = sum;
			// grads_ivar[z] = sum1;
			grads_sqrtvar[z] = -1*sum1/(epsilon+sigma[z]);
			grads_var[z] = 0.5 * (grads_sqrtvar[z]/sqrt(epsilon+sigma[z]));
		}
		
		for(int z = 0; z<in_size.z; z++){
			float sum = 0;
			for(int e=0; e<grad_next_layer.size.m; e++){
				for(int k=0; k<in_size.x; k++){
					for(int j=0; j<in_size.y; j++){
						// grads_xmu2(e,k,j,z) = (1.0/out.size.m)*grads_var[z]*2*(in(e,k,j,z) - u_mean[z]);
						grads_xmu2(e,k,j,z) = (grads_var[z]*2*(in(e,k,j,z) - u_mean[z]))/(out.size.m*out.size.x*out.size.y);
						grads_x1(e,k,j,z) = grads_xmu1(e,k,j,z) + grads_xmu2(e,k,j,z);
						sum += grads_x1(e,k,j,z);
					}
				}
			}
			grads_u_mean[z] = -sum; 
		}

		
		for(int z = 0; z<in_size.z; z++){
			for(int e=0; e<grad_next_layer.size.m; e++){
				for(int k=0; k<in_size.x; k++){
					for(int j=0; j<in_size.y; j++){
						// grads_in(e,k,j,z) = grads_x1(e,k,j,z) + (1.0/out.size.m)*grads_u_mean[z];
						grads_in(e,k,j,z) = grads_x1(e,k,j,z) + grads_u_mean[z]/(out.size.m*out.size.x*out.size.y);
					}
				}
			}
		}

		return grads_in;	
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "batch_norm2D" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}
	
	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json weights = {
			{ "epsilon", epsilon },
			{ "beta", beta },
			{ "gamma", gamma }
		};
		file << weights;
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weights;
		file >> weights;
		this->epsilon = weights["epsilon"];
		vector<float> beta = weights["beta"];
		vector<float> gamma = weights["gamma"];
		this->beta = beta;
		this->gamma = gamma;
		file.close();
	}

	void print_layer(){
		cout << "\n\n Batch Normalization Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
