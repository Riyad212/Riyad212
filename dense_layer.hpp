#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

/* Inkluderingsdirektiv: */
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

using namespace std;

struct dense_layer
{
	std::vector<double> output;
	std::vector<double> error;
	std::vector<double> bias;
	std::vector<std::vector<double>> weights;

	dense_layer(void) { }

	dense_layer(const std::size_t num_nodes,
		const std::size_t num_weights)
	{
		this->resize(num_nodes, num_weights);
		return;
	}

	~dense_layer(void)
	{
		this->clear();
		return;
	}

	/********************************************************************************
	* num_nodes: Returnerar antalet befintliga noder i angivet dense-lager.
	********************************************************************************/
	inline std::size_t num_nodes(void) const
	{
		return this->output.size();
	}

	/********************************************************************************
	* num_weights: Returnerar antalet vikter per nod i angivet dense-lager.
	********************************************************************************/
	inline std::size_t num_weights(void) const
	{
		return this->weights.size() ? this->weights[0].size() : 0;
	}

	/********************************************************************************
	* clear: Tömmer angiven vektor.
	********************************************************************************/
	void clear(void)
	{
		this->output.clear();
		this->error.clear();
		this->bias.clear();
		this->weights.clear();
		return;
	}

	void resize(const std::size_t num_nodes,
		const std::size_t num_weights)
	{
		this->output.resize(num_nodes, 0.0);
		this->error.resize(num_nodes, 0.0);
		this->bias.resize(num_nodes, 0.0);
		this->weights.resize(num_nodes, std::vector<double>(num_weights, 0.0));

		for (std::size_t i = 0; i < num_nodes; ++i)
		{
			this->bias[i] = this->get_random();

			for (std::size_t j = 0; j < num_weights; ++j)
			{
				this->weights[i][j] = this->get_random();
			}
		}

		return;
	}


	static void print(const std::vector<double>& data,
		std::ostream& ostream = std::cout,
		const std::size_t num_decimals = 1,
		const double threshold = 0.001)
	{
		ostream << std::fixed;

		for (auto& i : data)
		{
			ostream << std::setprecision(num_decimals) << get_rounded(i, threshold) << " ";
		}

		ostream << "\n";
		return;
	}

	void print(std::ostream& ostream = std::cout) const
	{
		ostream << "--------------------------------------------------------------------------------\n";
		ostream << "Number of nodes: " << this->num_nodes() << "\n";
		ostream << "Number of weights per node: " << this->num_weights() << "\n\n";

		ostream << "Output: ";
		this->print(this->output, ostream);

		ostream << "Error: ";
		this->print(this->error, ostream);

		ostream << "Bias: ";
		this->print(this->bias, ostream);

		ostream << "\nWeights:\n";

		for (std::size_t i = 0; i < this->num_nodes(); ++i)
		{
			ostream << "Node " << i + 1 << ": ";
			this->print(this->weights[i], ostream);
		}

		ostream << "--------------------------------------------------------------------------------\n\n";
		return;
	}

	void feedforward(const std::vector<double>& input)
	{
		for (std::size_t i = 0; i < this->num_nodes(); ++i)
		{
			auto sum = this->bias[i];

			for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
			{
				sum += input[j] * this->weights[i][j];
			}

			this->output[i] = this->tanh(sum);
		}

		return;
	}

	void backpropagate(const std::vector<double>& reference)
	{
		for (std::size_t i = 0; i < this->num_nodes() && i < reference.size(); ++i)
		{
			const auto dev = reference[i] - this->output[i];
			this->error[i] = dev * delta_tanh(this->output[i]);
		}

		return;
	}

	/********************************************************************************
	* backpropagate: Beräknar fel/avvikelser i angivet dolt lager via parametrar
	*                från nästa/efterföljande lager. OBS! Denna medlemsfunktion
	*                är avsedd enbart för dolda lager, se den alternativa
	*                medlemsfunktionen med samma namn för utgångslager.
	*
	*                - next_layer: Referens till nästa/efterföljande dense-lager.
	********************************************************************************/
	void backpropagate(const dense_layer& next_layer)
	{
		for (std::size_t i = 0; i < this->num_nodes(); ++i)
		{
			auto dev = 0.0;

			for (std::size_t j = 0; j < next_layer.num_nodes(); ++j)
			{
				dev += next_layer.error[j] * next_layer.weights[j][i];
			}

			this->error[i] = dev * this->delta_tanh(this->output[i]);
		}

		return;
	}

	void optimize(const std::vector<double>& input,
		const double learning_rate)
	{
		for (std::size_t i = 0; i < this->num_nodes(); ++i)
		{
			this->bias[i] += this->error[i] * learning_rate;

			for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
			{
				this->weights[i][j] += this->error[i] * learning_rate * input[j];
			}
		}

		return;
	}

private:
	/********************************************************************************
	* get_random: Returnerar ett randomiserat flyttal mellan 0.0 - 1.0.
	********************************************************************************/
	static inline double get_random(void)
	{
		return static_cast<double>(std::rand()) / RAND_MAX;
	}

	/********************************************************************************
	* relu: Implementerar en ReLU-funktion, där utsignalen y sätts till angiven
	*       summa sum om denna summa överstiger 0, annars sätts utsignalen till 0.
	*       Utsignalen y returneras efter beräkning. Om summan överstiger 0 så är
	*       aktuell nod aktiverad, annars är den inaktiverad:
	*
	*       sum > 0  => y = sum (noden aktiverad)
	*       sum <= 0 => y = 0   (noden inaktiverad)
	*
	*       - sum: Summan av nodens bias samt  indata (insignaler * vikter).
	********************************************************************************/
	static inline double tanh(const double sum)
	{
		return std::tanh(sum);
	}

	static inline double delta_tanh(const double output)
	{
		return 1 - std::pow(std::tanh(output), 2);
	}

	/********************************************************************************
	* get_rounded: Kontrollerar angivet flyttal och returnerar noll ifall detta
	*              ligger inom angivet intervall [-threshold, threshold].
	*              I praktiken gäller därmed att flyttalet jämförs med
	*              absolutbeloppet av angivet tröskelvärde. Annars om flyttalet
	*              i fråga inte ligger inom intervallet returneras detta tal.
	*
	*              - number   : Flyttalet som ska kontrolleras.
	*              - threshold: Tröskelvärdet som används för jämförelse.
	********************************************************************************/
	static double get_rounded(const double number,
		const double threshold = 0.001)
	{
		if (number > -threshold && number < threshold)
		{
			return 0;
		}
		else
		{
			return number;
		}
	}
};


#endif /* DENSE_LAYER_HPP_ */  #pragma once
