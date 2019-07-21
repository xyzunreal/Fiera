// Currently supports Momentum only
//TODO: Adding optimizers like adam etc.

#pragma once
#include "gradient_t.h"

#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.000

static float update_weight( float &w, gradient_t& grad, float multp, bool clip, float learning_rate)
{
	/**
	* 
	* Parameters
	* ----------
	* w : float
	* 		variable to be update
	*
	* grad : gradient_t
	* 		gradient of w (dw)
	* 
	* multp : float
	* 		
	* clip : bool
	* 		if need to clip weights
	*
	* learning_rate: float 
	*
	**/

	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	w -= learning_rate  * m * multp +
		 learning_rate * WEIGHT_DECAY * w;
		 
	// in case of binarization we need to clip weights
	if ( w < -1 and clip) 	  w = -1.0;
	else if (w > 1 and clip) w = 1.0;
	
	return w;
}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}

void clip_gradients(bool chk, float & gradient_value){
	//checking if gradients exceed 
	if(chk and gradient_value > 1e2)
		gradient_value = 1e2;

	else if(chk and gradient_value < -1e2)
		gradient_value = -1e2;
}