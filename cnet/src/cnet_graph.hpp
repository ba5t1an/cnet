/**
Wrapper class to ease use of neural network layers.

@file cnet_graph.hpp
@author Bastian Schoettle
*/


#ifndef CNET_GRAPH_HPP
#define CNET_GRAPH_HPP


#include <vector>
#include <memory>
#include <mutex>
#include "cnet_layer.hpp"
#include "cnet_vlogger.hpp"

namespace Cnet
{
	/*

	* Class Graph to wrap layer instances.
	*/
	class Graph
	{

	private:
		size_t _num_nodes;
		std::vector< std::shared_ptr<Layer > > _graph;
		bool _is_trainable;
		std::mutex _mutex;

		std::map<size_t, std::shared_ptr<Layer> > _layer_map;

		const static std::shared_ptr<Layer> get_layer_ptr(const size_t id, std::map<size_t, std::shared_ptr<Layer> > layer_map) {
			if (layer_map.find(id) != layer_map.end())
			{
				return layer_map[id];
			}
			return nullptr;
		}

	public:
		Graph() {
			this->_num_nodes = 0;
		}

		Graph(std::shared_ptr<Layer> graph) {
			this->_num_nodes = 0;
			this->_is_trainable = true;
			from_ptr(graph);
		}

		void from_ptr(std::shared_ptr<Layer> graph)
		{
			if (this->_num_nodes > 0)
			{
				this->_graph.clear();
				this->_num_nodes = 0;
				this->_is_trainable = true;
			}
			std::shared_ptr<Layer> current_layer = graph;
			do
			{
				if (current_layer->get_layer_id() == -1)
				{
					current_layer->set_layer_id(this->_num_nodes++);
				}
				this->_graph.push_back(current_layer);
                std::shared_ptr<Layer> skip_layer = current_layer->get_skip_layer();
                if (skip_layer)
                {
                    skip_layer->set_bwd_skip(current_layer);
                }
				current_layer = current_layer->get_previous_layer();

			} while (current_layer->has_previous());
			current_layer->set_layer_id(this->_num_nodes++);
            std::shared_ptr<Layer> skip_layer = current_layer->get_skip_layer();
            if (skip_layer)
            {
                skip_layer->set_bwd_skip(current_layer);
            }
			this->_graph.push_back(current_layer);
			std::reverse(std::begin(this->_graph), std::end(this->_graph));
		}

		std::vector<std::shared_ptr<Layer> >* get_nodes()
		{
			return &this->_graph;
		}

		bool is_trainable()
		{
			return _is_trainable;
		}

		void set_trainable(bool trainable)
		{
			this->_is_trainable = trainable;
		}

		void add_layer(std::shared_ptr<Layer> layer)
		{
			if (this->_num_nodes == 0)
			{
				layer->set_layer_id(this->_num_nodes++);
				this->_graph.push_back(layer);
			}
			else
			{
				layer->set_previous_layer(this->_graph[this->_num_nodes - 1]);
				layer->set_layer_id(this->_num_nodes++);
				this->_graph.push_back(std::move(layer));
			}
		}

		size_t get_num_layers()
		{
			return this->_num_nodes;
		}

		std::shared_ptr<Layer> get_layer_by_idx(size_t idx)
		{
			return this->_graph[idx];
		}

		void forward(MatrixRm* input, MatrixRm* output, bool training = false)
		{
            std::unique_lock<std::mutex> lock(_mutex);
            this->_graph[0]->forward(input);
			for (size_t i = 1; i < this->_num_nodes - 1; ++i)
			{
				this->_graph[i]->forward(training);
			}
			this->_graph[this->_num_nodes - 1]->forward(training);
			*output = (*this->_graph[this->_num_nodes - 1]->get_output());
		}

        MatrixRm* fwd(MatrixRm* input, bool training = false)
        {
            std::unique_lock<std::mutex> lock(_mutex);
            this->_graph[0]->forward(input);
            for (size_t i = 1; i < this->_num_nodes - 1; ++i)
            {
                this->_graph[i]->forward(training);
            }
            this->_graph[this->_num_nodes - 1]->forward(training);
            return this->_graph[this->_num_nodes - 1]->get_output();
        }

		//TODO: Implement...obviously
		void forward_argmax(MatrixRm* input, unsigned int *max_idx)
		{
			(void) input;
			(void) max_idx;
		}

		//TODO: Implement...obviously
		void forward_argmax(MatrixRm* input, MatrixRm& output)
		{
			(void) input;
			(void) output;
		}

		void sync_weights(Graph* g)
		{
			for (size_t i = 0; i < this->_num_nodes; ++i)
			{
				*this->_graph[i]->get_weights() = *(*g->get_nodes())[i]->get_weights();
				*this->_graph[i]->get_bias() = *(*g->get_nodes())[i]->get_bias();
			}
		}

		/*
		Sums gradients 
		*/
		void gather_gradients(Graph *g)
		{
			std::unique_lock<std::mutex> lock(_mutex);
			for (size_t i = 0; i < this->_num_nodes; ++i)
			{
				*this->_graph[i]->get_gradients() += *(*g->get_nodes())[i]->get_gradients();
				*this->_graph[i]->get_bias_gradients() += *(*g->get_nodes())[i]->get_bias_gradients();
			}
		}

		/*
		* Sums and resets gradients
		*/
		void gather_and_reset_gradients(Graph *g)
		{
			std::unique_lock<std::mutex> lock(_mutex);
			for (size_t i = 0; i < this->_num_nodes; ++i)
			{
				*this->_graph[i]->get_gradients() += *(*g->get_nodes())[i]->get_gradients();
				*this->_graph[i]->get_bias_gradients() += *(*g->get_nodes())[i]->get_bias_gradients();
				//Reset gradients
				(*g->get_nodes())[i]->get_gradients()->setZero();
				(*g->get_nodes())[i]->get_bias_gradients()->setZero();
			}
		}

		/*
		 * TODO: Add activations to map
		 */
		void clone(Graph* clone)
		{
			std::map<size_t, std::shared_ptr<Layer> > layer_map;
			std::shared_ptr<InputLayer> input = std::dynamic_pointer_cast<InputLayer>(_graph[0]);
			std::shared_ptr<Layer> prev = std::make_shared<InputLayer>((size_t) sqrt(input->get_output_size()), input->get_output_channels());
			layer_map[prev->get_layer_id()] = prev;
			for (size_t i = 1; i < this->_num_nodes; i++)
			{
				if (_graph[i]->get_layer_type() == CONV2D_LAYER)
				{
					std::shared_ptr<Conv2dLayer> layer = std::dynamic_pointer_cast<Conv2dLayer>(_graph[i]);
					ConvolutionParams params = layer->get_conv_params();
					PaddingPolicy pad = VALID;
					if (params.padding() > 0)
					{
						pad = SAME;
					}
					prev = std::make_shared<Conv2dLayer>(prev, layer->get_num_kernels(), params.kernel_width(), params.stride(), pad, Activation::create_activation(layer->get_activation()->get_type()));
					*prev->get_weights() = *layer->get_weights();
					*prev->get_bias() = *layer->get_bias();
					layer_map[prev->get_layer_id()] = prev;
				}
				else if (_graph[i]->get_layer_type() == FC_LAYER)
				{
					std::shared_ptr<FcLayer> layer = std::dynamic_pointer_cast<FcLayer>(_graph[i]);
					prev = std::make_shared<FcLayer>(prev, layer->get_output_size(), Activation::create_activation(layer->get_activation()->get_type()));
					*prev->get_weights() = *layer->get_weights();
					*prev->get_bias() = *layer->get_bias();
					layer_map[prev->get_layer_id()] = prev;
				}
				else if (_graph[i]->get_layer_type() == CONV2D_CC_LAYER)
				{
					std::shared_ptr<Conv2dConcatLayer> layer = std::dynamic_pointer_cast<Conv2dConcatLayer>(_graph[i]);
					ConvolutionParams params = layer->get_conv_params();
					std::shared_ptr<Layer> skip_layer = get_layer_by_idx(layer->get_crop_layer()->get_layer_id());
					prev = std::unique_ptr<Conv2dConcatLayer>(new Conv2dConcatLayer(prev, skip_layer,layer->get_num_kernels(), params.kernel_width(), params.stride(), Activation::create_activation(layer->get_activation()->get_type())));
                    skip_layer->set_bwd_skip(prev);
					*prev->get_weights() = *layer->get_weights();
					*prev->get_bias() = *layer->get_bias();
					layer_map[prev->get_layer_id()] = prev;
				}
				else if (_graph[i]->get_layer_type() == CONV2D_T_LAYER)
				{
					std::shared_ptr<Conv2dTransposeLayer> layer = std::dynamic_pointer_cast<Conv2dTransposeLayer>(_graph[i]);
					ConvolutionParams params = layer->get_conv_params();
					prev = std::make_shared<Conv2dTransposeLayer>(prev, layer->get_num_kernels(), params.kernel_width(), params.stride(), Activation::create_activation(layer->get_activation()->get_type()));
					*prev->get_weights() = *layer->get_weights();
					*prev->get_bias() = *layer->get_bias();
					layer_map[prev->get_layer_id()] = prev;
				}
				else if (_graph[i]->get_layer_type() == DROP_LAYER)
				{
					std::shared_ptr<DropoutLayer> layer = std::dynamic_pointer_cast<DropoutLayer>(_graph[i]);
					prev = std::make_shared<DropoutLayer>(prev, layer->get_dropout_prob());
					layer_map[prev->get_layer_id()] = prev;
				}
				else if (_graph[i]->get_layer_type() == MAXPOOL2D_LAYER)
				{
					std::shared_ptr<MaxPool2dLayer> layer = std::dynamic_pointer_cast<MaxPool2dLayer>(_graph[i]);
					prev = std::make_shared<MaxPool2dLayer>(prev, layer->get_window_size(), layer->get_stride());
					layer_map[prev->get_layer_id()] = prev;
				}
			}
			clone->from_ptr(prev);
		}


		MatrixRm forward(MatrixRm* input, bool training = false)
		{
            this->_graph[0]->forward(input);
			for (size_t i = 1; i < this->_num_nodes - 1; ++i)
			{
				this->_graph[i]->forward(training);
			}
			this->_graph[this->_num_nodes - 1]->forward(training);
			return (*this->_graph[this->_num_nodes - 1]->get_output());
		}

		void visit(MatrixRm* input, Visitor* visitor)
		{
            this->_graph[0]->forward(input);
			this->_graph[0]->accept(visitor);
			for (size_t i = 1; i < this->_num_nodes - 1; ++i)
			{
				this->_graph[i]->forward();
				this->_graph[i]->accept(visitor);
				this->_graph[i]->reset_layer();
			}
			this->_graph[this->_num_nodes - 1]->forward();
			this->_graph[this->_num_nodes - 1]->accept(visitor);
			this->_graph[this->_num_nodes - 1]->reset_layer();
		}



		void backward(MatrixRm* deltas)
		{
            MatrixRm* layer_deltas = deltas;
			for (size_t i = (this->_num_nodes - 1); i > 0; --i)
			{
				this->_graph[i]->backward(layer_deltas);
				layer_deltas = this->_graph[i]->get_dx();
			}
		}

		size_t get_output_cols()
		{
			return _graph[this->_num_nodes - 1]->get_output()->cols();
		}

		size_t get_output_rows()
		{
			return _graph[this->_num_nodes - 1]->get_output()->rows();
		}

		void reset_gradients()
		{
			for (size_t i = this->_num_nodes - 1; i > 0; --i)
			{
				this->_graph[i]->reset_gradients();
			}
		}

        size_t get_input_size()
        {
            return _graph[0]->get_output_size();
        }

        size_t get_input_channels()
        {
            return _graph[0]->get_output_channels();
        }

        size_t get_output_size()
        {
            return _graph[_graph.size()-1]->get_output_size();
        }

        size_t get_output_channels()
        {
            return _graph[_graph.size()-1]->get_output_channels();
        }

	};

}

#endif