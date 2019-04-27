#pragma once
#include "../Libraries/json.hpp"
using json = nlohmann::json;
struct point_t
{
	int m, x, y, z;
	void from_json( json j){
		// TODO: 'assert 'j' is of form [int, int, int, int]
		m = j[0];
		x = j[1];
		y = j[2];
		z = j[3];
	}
};
typedef point_t tdsize;