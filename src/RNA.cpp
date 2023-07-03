
#include "RNA.hpp"

#include <iostream>

RNA::RNA(size_t i_nodes, size_t h_nodes, size_t o_nodes) : i_nodes(i_nodes), h_nodes(h_nodes), o_nodes(o_nodes)
{
	ih_bias.alloc(this->h_nodes, 1);
	ih_bias.rnd(-1, 1);
	ho_bias.alloc(this->o_nodes, 1);
	ho_bias.rnd(-1, 1);

	ih_weights.alloc(this->h_nodes, this->i_nodes);
	ih_weights.rnd(-1, 1);
	ho_weights.alloc(this->o_nodes, this->h_nodes);
	ho_weights.rnd(-1, 1);
}