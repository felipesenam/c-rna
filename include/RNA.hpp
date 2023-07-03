
#ifndef B9B71DAF_80CE_41D0_BAF3_0F7D88E63E32
#define B9B71DAF_80CE_41D0_BAF3_0F7D88E63E32

#include <cstddef>

#include "Matrix.hpp"

class RNA
{
private:
	size_t i_nodes, h_nodes, o_nodes;

	Matrix<double> ih_bias;
	Matrix<double> ho_bias;

	Matrix<double> ih_weights;
	Matrix<double> ho_weights;

public:
	double learning_rate = 0.1;

	RNA(size_t i_nodes, size_t h_nodes, size_t o_nodes);

	template <typename T>
	void train(const Matrix<T> &input, const Matrix<T> &target)
	{
		Matrix<T> hidden(Matrix<T>::sigmoid((ih_weights * input) + ih_bias));

		Matrix<T> output(Matrix<T>::sigmoid((ho_weights * hidden) + ho_bias));

		Matrix<T> output_error(target - output);

		Matrix<T> gradient_O(Matrix<T>::hadamard(output_error, Matrix<T>::dsigmoid(output)) * learning_rate);
		ho_bias = ho_bias + gradient_O;

		Matrix<T> ho_deltas(gradient_O * Matrix<T>::transpose(hidden));
		ho_weights = ho_weights + ho_deltas;

		Matrix<T> gradient_H((Matrix<T>::hadamard(Matrix<T>::transpose(ho_weights) * output_error, Matrix<T>::dsigmoid(hidden))) * learning_rate);
		ih_bias = ih_bias + gradient_H;

		Matrix<T> ih_deltas(gradient_H * Matrix<T>::transpose(input));
		ih_weights = ih_weights + ih_deltas;
	}

	template <typename T>
	Matrix<T> predict(const Matrix<T> &input)
	{
		Matrix<T> hidden(Matrix<T>::sigmoid((ih_weights * input) + ih_bias));
		Matrix<T> output(Matrix<T>::sigmoid((ho_weights * hidden) + ho_bias));

		return output;
	}
};

#endif /* B9B71DAF_80CE_41D0_BAF3_0F7D88E63E32 */
