#pragma once 

#include "dense_layer.hpp"

using namespace std;

/********************************************************************************
* ann: Denna klassen för implimitering av nerualt nätver används för ett 
*      ingångslager, dolt lager samt ett utgånglager med slumpmässig antal noder. 
*      Träningsdata kan gå föörbi vi vektorer. När träningen har genomförsts så 
*      kan prediktion med utskrift genomföras med godtycklig indata eller med 
*      indata från förekommande träningsuppsättningar. 
********************************************************************************/

class ann {

private:
	vector<dense_layer> hidden_layers;
	dense_layer output_layer;
	vector<vector<double>> button_in;
	vector<vector<double>> diod_out;
	vector<size_t> train_order;

	/********************************************************************************
   * feedforward: Används för att berkna nya utsignaler för samtliga noder i det 
   *              neurala nätverk via angiven indata. 
   *             
   *
   *              - input: används till referns till vekttor innehållande ny indata
   ********************************************************************************/

	void feedforward(const vector<double>& input) {
		this->hidden_layers[0].feedforward(input);

		for (std::size_t i = 1; i < hidden_layers.size(); ++i)
		{
			this->hidden_layers[i].feedforward(this->hidden_layers[i - 1].output);
		}
		this->output_layer.feedforward(this->last_hidden_layer().output);
		return;
	}

	/********************************************************************************
   * backpropagate: Används för att beräkna antal fel för samtliga noder i angiven
   *                neuralt nätverk via jämförelse med referensdata innehållande 
   *                korrekta utsignaler, som sedan jämförs med det predikterade 
   *                 utsignalen.                
   *
   *                - reference: : Referens till vektor innehållande korrekta värden.
   ********************************************************************************/

	void backpropagate(const vector<double>& reference) {
		this->output_layer.backpropagate(reference);
		last_hidden_layer().backpropagate(this->output_layer);

		for (std::size_t i = hidden_layers.size() - 1; i > 0; --i)
		{
			this->hidden_layers[i - 1].backpropagate(this->hidden_layers[i]);
		}
	}

	/********************************************************************************
   * optimize:
   *           
   *           
   *
   *           
   *           - learning_rate:
   *                            
   ********************************************************************************/

	void optimize(const vector<double>& input, const double learning_rate) {
		first_hidden_layer().optimize(input, learning_rate);

		for (size_t i = 1; i < this->hidden_layers.size(); i++) {
			this->hidden_layers[i].optimize(this->hidden_layers[i - 1].output, learning_rate);
		}

		output_layer.optimize(last_hidden_layer().output, learning_rate);
	}

	dense_layer& first_hidden_layer(void) {
		return this->hidden_layers[0];
	}

	dense_layer& last_hidden_layer(void) {
		return this->hidden_layers[this->hidden_layers.size() - 1];
	}

	void shuffel(void) {
		
		for (size_t i = 0; i < this->train_order.size(); i++)
		{
			auto r = rand() % this->train_order.size();
			auto copy = train_order[i];
			train_order[i] = train_order[r];
			train_order[r] = copy;

		}
		return;
	}

public:

	ann(void) { cout << " Multi is created"; }


	ann(const size_t num_inputs,
		const size_t num_hidden_layers,
		const size_t num_hidden_nodes,
		const size_t num_outputs) {
		this->init(num_inputs, num_hidden_layers, num_hidden_nodes, num_outputs);
	}



	~ann(void) {
		this->clear();
		return;
	}

	const vector<dense_layer>& get_hidden_layers(void) const {
		return this->hidden_layers;
	}

	const dense_layer& get_output_layer(void) const {
		return this->output_layer;
	}

	const vector<vector<double>>& get_button_in(void) const {
		return this->button_in;
	}

	const vector<vector<double>>& get_diod_out(void) const {
		return this->diod_out;
	}



	void init(const size_t num_inputs,
		const size_t num_hidden_layers,
		const size_t num_hidden_nodes,
		const size_t num_outputs) {

		dense_layer first_hidden(num_hidden_nodes, num_inputs);
		hidden_layers.push_back(first_hidden);

		this->output_layer.resize(num_outputs, num_hidden_nodes);

		for (size_t i = 1; i < this->hidden_layers.size(); i++) {
			dense_layer hidden_layer(num_hidden_nodes, num_hidden_nodes);
			hidden_layers.push_back(hidden_layer);
		}
	}

	void clear(void) {

		this->hidden_layers.clear();
		this->output_layer.clear();
		this->button_in.clear();
		this->diod_out.clear();
		this->train_order.clear();
		return;
	}

	void set_training_data(const vector<vector<double>>& button_in,
		const vector<vector<double>>& diod_out) {
		this->button_in = button_in;
		this->diod_out = diod_out;

		if (button_in.size() != diod_out.size()) {
			size_t num_sets = 0;

			if (button_in.size() < diod_out.size()) {
				num_sets = button_in.size();
			}
			else {
				num_sets = diod_out.size();
			}

			this->button_in.resize(num_sets);
			this->diod_out.resize(num_sets);
		}

		this->train_order.resize(this->button_in.size());

		for (size_t i = 0; i < this->train_order.size(); i++)
		{
			this->train_order[i] = i;
		}

		return;
	}

	void train(const size_t num_epochs,
		const double learning_rate) {
		for (size_t i = 0; i < num_epochs; i++) {
			this->shuffel();

			for (size_t j = 0; j < this->train_order.size(); j++) {
				const auto index = this->train_order[j];
				const auto& input = this->button_in[index];
				const auto& reference = this->diod_out[index];

				this->feedforward(input);
				this->backpropagate(reference);
				this->optimize(input, learning_rate);
			}
		}

		return;
	}

	const vector<double>& predict(const vector<double>& input) {
		this->feedforward(input);
		return this->output_layer.output;
	}

	void print(const vector<vector<double>>& input,
		const size_t num_decimals = 1,
		ostream& ostream = cout,
		const double threshold = 0.001) {

		const auto& end = input[input.size() - 1];
		if (input.size() == 0) return;

		ostream << "-----------------------------------------------------------------\n";

		for (auto& i : input) {
			ostream << " Input: ";
			dense_layer::print(i, ostream, num_decimals, threshold);


			ostream << "Output: ";
			dense_layer::print(this->predict(i), ostream, num_decimals, threshold);

			if (&i < &end) ostream << "\n";
		}

		ostream << "-----------------------------------------------------------------\n\n";
	}




public:

	/********************************************************************************
	* print: Genomför prediktion med samtliga befintliga träningsuppsättningars
	*        indata och skriver ut predikterad utdata via angiven utström, där
	*        standardutenheten std::cout används som default för utskrift i
	*        terminalen. Som default sker utskrift med en decimal. Värden mellan
	*        -0.001 samt 0.001 avrundas till noll vid utskrift.
	*
	*        - num_decimals: Antalet decimaler vid utskrift (default = 1).
	*        - ostream     : Referens till godtycklig utström (default = std::cout).
	*        - threshold   : Tröskelvärde som används för att avrunda tal nära
	*                        noll (default = 0.001).
	********************************************************************************/
	void print(const size_t num_decimals = 1,
		ostream& ostream = std::cout,
		const double threshold = 0.001)
	{
		this->print(this->button_in, num_decimals, ostream, threshold);
		return;
	}

};