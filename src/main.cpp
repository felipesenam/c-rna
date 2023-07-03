#include <iostream>
#include <array>
#include <random>
#include <ctime>

#include "RNA.hpp"

int main(/*int argc, char *argv[]*/)
{
	RNA nn(1, 4, 1);
	Matrix<double> input(50, 1);
	Matrix<double> output(50, 1);

	std::default_random_engine gen(time(0));
	std::uniform_int_distribution<> dist(0, input.lines() - 1);

	while (1)
	{
		for (int j = 0; j < 10; ++j)
			for (size_t i = 0; i < 1000000; ++i)
			{
				size_t index = dist(gen);
				nn.train(input.line(index), output.line(index));
			}
		std::cout << "enter a number: ";

		Matrix<double> p(1, 1);
		std::cin >> p.at(0, 0);
		if (std::cin.fail())
		{
			std::cin.clear();
			break;
		}
		std::cout << p << std::endl;
		std::cout << nn.predict(p) << std::endl;
	}

	return 0;
}