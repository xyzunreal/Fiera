#pragma once
#include "gradient_t.h"

#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.000

static float update_weight( float &w, gradient_t& grad, float multp, bool clip, float learning_rate)
{

	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	w -= learning_rate  * m * multp +
		 learning_rate * WEIGHT_DECAY * w;
		 
		 if ( w < -1 and clip) 	  w = -1.0;
		 else if (w > 1 and clip) w = 1.0;
	return w;

}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}

void clip_gradients(bool chk, float & gradient_value){

	if(chk and gradient_value > 1e2)
		gradient_value = 1e2;

	else if(chk and gradient_value < -1e2)
		gradient_value = -1e2;
}