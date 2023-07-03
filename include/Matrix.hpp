
#ifndef E5D9B508_E38D_45C8_8BC9_A39F41936F5E
#define E5D9B508_E38D_45C8_8BC9_A39F41936F5E

#include <sstream>
#include <exception>
#include <typeinfo>
#include <initializer_list>
#include <random>
#include <iostream>
#include <ctime>
#include <cmath>

#ifndef signature
#define signature std::string(typeid(*this).name()) + "::" + __func__
#endif

#ifndef ssignature
#define ssignature(x) std::string(x) + "::" + __func__
#endif

template <typename T>
class Matrix
{
private:
	size_t nlin = 0;
	size_t ncol = 0;

	T **values = nullptr;

public:
	Matrix() {}
	Matrix(size_t lines, size_t columns)
	{
		alloc(lines, columns);
	}
	Matrix(const Matrix<T> &matrix)
	{
		*this = matrix;
	}
	Matrix(std::initializer_list<std::initializer_list<T>> list)
	{
		alloc(list.size(), list.begin()->size());

		size_t i = 0;
		for (auto line = list.begin(); line != list.end(); ++line)
		{
			if (line->size() > ncol)
				throw irregular_initializer_list(signature);

			size_t j = 0;
			for (auto value = line->begin(); value != line->end(); ++value)
			{
				values[i][j] = *value;
				j++;
			}
			i++;
		}
	}

	~Matrix()
	{
		erase();
	}

	struct out_of_bounds : std::exception
	{
		const std::string func;
		out_of_bounds(const std::string &func) : func(func) {}
		const char *what() const noexcept
		{
			return func.c_str();
		}
	};

	struct irregular_initializer_list : std::exception
	{
		const std::string func;
		irregular_initializer_list(const std::string &func) : func(func) {}
		const char *what() const noexcept
		{
			return func.c_str();
		}
	};
	struct invalid_matrix_operation : std::exception
	{
		const std::string func;
		invalid_matrix_operation(const std::string &func) : func(func) {}
		const char *what() const noexcept
		{
			return func.c_str();
		}
	};

	Matrix<T> line(size_t line)
	{
		if (line >= nlin)
			throw out_of_bounds(signature);

		Matrix<T> newMatrix(ncol, 1);
		for (size_t i = 0; i < ncol; ++i)
			newMatrix.values[i][0] = values[line][i];

		return newMatrix;
	}
	T &at(size_t line, size_t column)
	{
		if (line >= nlin || column >= ncol)
			throw out_of_bounds(signature);

		return values[line][column];
	}
	const T &get(size_t line, size_t column) const
	{
		if (line >= nlin || column >= ncol)
			throw out_of_bounds(signature);

		return values[line][column];
	}

	void alloc(size_t lines, size_t columns)
	{
		erase();
		ncol = columns;
		nlin = lines;

		values = new T *[nlin];
		for (size_t i = 0; i < nlin; ++i)
			values[i] = new T[ncol];
	}
	void erase()
	{
		if (values == nullptr)
			return;

		for (size_t i = 0; i < nlin; ++i)
			delete[] values[i];
		delete[] values;

		ncol = 0;
		nlin = 0;
		values = nullptr;
	}

	void rnd(double from, double to)
	{
		static std::default_random_engine gen(time(0));

		std::uniform_real_distribution<> dist(from, to);
		for (size_t i = 0; i < nlin; ++i)
			for (size_t j = 0; j < ncol; ++j)
				values[i][j] = static_cast<T>(dist(gen));
	}

	static Matrix<T> sigmoid(const Matrix<T> &matrix)
	{
		Matrix<T> newMatrix(matrix.lines(), matrix.columns());
		for (size_t i = 0; i < matrix.lines(); ++i)
			for (size_t j = 0; j < matrix.columns(); ++j)
				newMatrix.at(i, j) = 1 / (1 + exp(-matrix.get(i, j)));

		return newMatrix;
	}
	void sigmoid()
	{
		for (size_t i = 0; i < nlin; ++i)
			for (size_t j = 0; j < ncol; ++j)
				values[i][j] = 1 / (1 + exp(-values[i][j]));
	}

	static Matrix<T> dsigmoid(const Matrix<T> &matrix)
	{
		Matrix<T> newMatrix(matrix.lines(), matrix.columns());
		for (size_t i = 0; i < matrix.lines(); ++i)
			for (size_t j = 0; j < matrix.columns(); ++j)
				newMatrix.at(i, j) = matrix.get(i, j) * (1 - matrix.get(i, j));

		return newMatrix;
	}
	void dsigmoid()
	{
		for (size_t i = 0; i < nlin; ++i)
			for (size_t j = 0; j < ncol; ++j)
				values[i][j] = values[i][j] * (1 - values[i][j]);
	}

	static Matrix<T> hadamard(const Matrix<T> &matrix1, const Matrix<T> &matrix2)
	{
		if (matrix1.lines() != matrix2.lines() || matrix1.columns() != matrix2.columns())
			throw invalid_matrix_operation(ssignature("Matrix"));

		Matrix newMatrix(matrix1.lines(), matrix1.columns());
		for (size_t i = 0; i < matrix1.lines(); ++i)
			for (size_t j = 0; j < matrix1.columns(); ++j)
				newMatrix.at(i, j) = matrix1.get(i, j) * matrix2.get(i, j);

		return newMatrix;
	}
	Matrix<T> hadamard(const Matrix<T> &matrix)
	{
		if (nlin != matrix.nlin || ncol != matrix.ncol)
			throw invalid_matrix_operation(signature);

		Matrix newMatrix(nlin, ncol);
		for (size_t i = 0; i < nlin; ++i)
			for (size_t j = 0; j < ncol; ++j)
				newMatrix.values[i][j] = values[i][j] * matrix.values[i][j];

		return newMatrix;
	}

	static Matrix<T> transpose(const Matrix<T> &matrix)
	{
		Matrix newMatrix(matrix.columns(), matrix.lines());
		for (size_t i = 0; i < newMatrix.lines(); ++i)
			for (size_t j = 0; j < newMatrix.columns(); ++j)
				newMatrix.at(i, j) = matrix.get(j, i);

		return newMatrix;
	}

	size_t lines() const noexcept
	{
		return nlin;
	}
	size_t columns() const noexcept
	{
		return ncol;
	}
	size_t size() const noexcept
	{
		return nlin * ncol;
	}

	Matrix<T> &operator=(const Matrix<T> &matrix)
	{
		alloc(matrix.nlin, matrix.ncol);
		for (size_t i = 0; i < nlin; ++i)
			for (size_t j = 0; j < ncol; ++j)
				values[i][j] = matrix.values[i][j];

		return *this;
	}

	Matrix<T> operator*(const Matrix<T> &matrix) const
	{
		if (ncol != matrix.nlin)
			throw invalid_matrix_operation(signature);

		Matrix<T> newMatrix(nlin, matrix.ncol);
		for (size_t i = 0; i < newMatrix.nlin; ++i)
		{
			for (size_t j = 0; j < newMatrix.ncol; ++j)
			{
				newMatrix.values[i][j] = 0;
				for (size_t k = 0; k < matrix.nlin; ++k)
				{
					newMatrix.values[i][j] += values[i][k] * matrix.values[k][j];
				}
			}
		}
		return newMatrix;
	}

	Matrix<T> operator*(double esc) const
	{
		Matrix<T> newMatrix(*this);
		for (size_t i = 0; i < nlin; ++i)
			for (size_t j = 0; j < ncol; ++j)
				newMatrix.values[i][j] = values[i][j] * esc;

		return newMatrix;
	}

	Matrix<T> operator+(const Matrix<T> &matrix) const
	{
		if (nlin != matrix.nlin || ncol != matrix.ncol)
			throw invalid_matrix_operation(signature);

		Matrix<T> newMatrix(nlin, ncol);
		for (size_t i = 0; i < newMatrix.nlin; ++i)
			for (size_t j = 0; j < newMatrix.ncol; ++j)
				newMatrix.values[i][j] = values[i][j] + matrix.values[i][j];

		return newMatrix;
	}

	Matrix<T> operator-(const Matrix<T> &matrix) const
	{
		if (nlin != matrix.nlin || ncol != matrix.ncol)
			throw invalid_matrix_operation(signature);

		Matrix<T> newMatrix(nlin, ncol);
		for (size_t i = 0; i < newMatrix.nlin; ++i)
			for (size_t j = 0; j < newMatrix.ncol; ++j)
				newMatrix.values[i][j] = values[i][j] - matrix.values[i][j];

		return newMatrix;
	}

	friend std::ostream &operator<<(std::ostream &o, const Matrix<T> &matrix)
	{
		o << matrix.str();
		return o;
	}

	std::string str() const
	{
		std::ostringstream stream;
		stream << "[";
		for (size_t i = 0; i < nlin; ++i)
		{
			stream << "[";
			for (size_t j = 0; j < ncol; ++j)
			{
				stream << values[i][j];
				if (j + 1 < ncol)
					stream << ", ";
			}
			stream << "]";
			if (i + 1 < nlin)
				stream << ", ";
		}
		stream << "]";

		return stream.str();
	}
};

#endif /* E5D9B508_E38D_45C8_8BC9_A39F41936F5E */
