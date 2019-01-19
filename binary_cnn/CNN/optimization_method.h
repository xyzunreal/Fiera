#pragma once
#include "gradient_t.h"

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

static float update_weight( float &w, gradient_t& grad, float multp = 1, bool clip=false )
{
	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	w -= LEARNING_RATE  * m * multp +
		 LEARNING_RATE * WEIGHT_DECAY * w;

	if ( w < -1 and clip) 
		w = -1;
	else if (w > 1 and clip) 
		w = 1;

	return w;

}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}