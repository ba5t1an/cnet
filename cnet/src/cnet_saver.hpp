/**
Implementation of serval Saver classes wich allow storing graphs on fs.

@file cnet_saver.hpp
@author Bastian Schoettle EN RC PREC
*/


#ifndef CNET_SAVER_HPP
#define CNET_SAVER_HPP

#include <string>
#include <map>
#include <memory>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include "cnet_graph.hpp"
#include "cnet_layer.hpp"
#include "cnet_activation.hpp"

namespace Cnet
{

	/*
	* Base class to store layers.
	*/
	class Saver
	{

	private:
		std::map<ActivationType, std::shared_ptr<Activation> > _activation_map;

	protected:
		std::map<size_t, std::shared_ptr<Layer> > _layer_map;
		std::string _model_path;
		/*
		* Function to retrive already created instances of activations from a map.
		*/
		const std::shared_ptr<Activation> get_activation_ptr(const ActivationType type) {
			if (this->_activation_map.find(type) != this->_activation_map.end())
			{
				return this->_activation_map[type];
			}
			this->_activation_map[type] = Activation::create_activation(type);
			return this->_activation_map[type];
		}

		/*
		* Function to retrive already created instances of activations from a map.
		*/
		const std::shared_ptr<Layer> get_layer_ptr(const size_t id) {
			if (this->_layer_map.find(id) != this->_layer_map.end())
			{
				return this->_layer_map[id];
			}
			return nullptr;
		}

	public:

		/*
		* Constructor for saver. Takes model_path as arg to store or restore a model from.
		*/
		Saver(std::string model_path)
		{
			this->_model_path = model_path;
		}

		/*
		* Store method to actually store layer in whatever file type is used by the inheriting instance of solver.  
		*/
		virtual void store(Graph& graph, size_t epoch = 0, size_t batch = 0, double valid_acc = 0.f) = 0;

		/*
		* Restore method to load a model by a specified file type. 
		*/
		virtual void restore(Graph* graph) = 0;

		std::string get_model_path()
		{
			return this->_model_path;
		}

	};

	/*
	* Class JsonSaver stores and restores models in .json format.
	*/
	class JsonSaver : public Saver
	{
	public:

		/*
		* Constructor for JsonSaver class.
		*/
		JsonSaver(std::string model_path) : Saver(model_path)
		{
		}

		/*
		* Stores layer depending on type. Any new Layer must be added here...
		*/
		void store_layer(boost::property_tree::ptree& root, std::shared_ptr<Layer> current_layer, size_t idx, size_t& num_params)
		{
			if (current_layer->get_layer_type() == CONV2D_LAYER)
			{
				store_conv2d(root, current_layer, idx, num_params);
			}
			else if (current_layer->get_layer_type() == CONV2D_T_LAYER)
			{
				store_conv2d_t(root, current_layer, idx, num_params);
			}
			else if (current_layer->get_layer_type() == FC_LAYER)
			{
				store_fc_layer(root, current_layer, idx, num_params);
			}
			else if (current_layer->get_layer_type() == INPUT_LAYER)
			{
				store_input(root, current_layer, idx);
			}
			else if (current_layer->get_layer_type() == MAXPOOL2D_LAYER)
			{
				store_maxpool2d(root, current_layer, idx);
			}
			else if (current_layer->get_layer_type() == CONV2D_CC_LAYER)
			{
				store_conv2d_cc(root, current_layer, idx, num_params);
			}
		}

		/*
		* Stores conv2d layer.
		*/
		void store_conv2d(boost::property_tree::ptree& root, const std::shared_ptr<Layer>& current_layer, const size_t idx, size_t& num_params)
		{
			std::shared_ptr<Conv2dLayer> conv2d = std::dynamic_pointer_cast<Conv2dLayer>(current_layer);
			root.put("model.layer." + std::to_string(idx) + ".type", std::to_string(conv2d->get_layer_type()));
			root.put("model.layer." + std::to_string(idx) + ".id", std::to_string(conv2d->get_layer_id()));
			root.put("model.layer." + std::to_string(idx) + ".filter_count", std::to_string(conv2d->get_weights()->rows()));
			ConvolutionParams  conv_params = conv2d->get_conv_params();
			root.put("model.layer." + std::to_string(idx) + ".filter_dim", std::to_string(conv_params.kernel_width()));
			root.put("model.layer." + std::to_string(idx) + ".stride", std::to_string(conv_params.stride()));
			if (conv_params.padding() > 0)
			{
				root.put("model.layer." + std::to_string(idx) + ".pad", std::to_string(SAME));
			}
			else
			{
				root.put("model.layer." + std::to_string(idx) + ".pad", std::to_string(VALID));
			}
			root.put("model.layer." + std::to_string(idx) + ".activation", std::to_string(conv2d->get_activation()->get_type()));
			boost::property_tree::ptree weight_matrix;
			for (int j = 0; j < conv2d->get_weights()->rows(); j++)
			{
				boost::property_tree::ptree weight_row;
				for (int k = 0; k < conv2d->get_weights()->cols(); k++)
				{
					boost::property_tree::ptree weight;
					weight.put_value((*conv2d->get_weights())(j, k));
					weight_row.push_back(std::make_pair("", weight));
					++num_params;
				}
				weight_matrix.push_back(std::make_pair(std::to_string(j), weight_row));
			}
			root.add_child("model.layer." + std::to_string(idx) + ".weights", weight_matrix);
			boost::property_tree::ptree bias_matrix;
			boost::property_tree::ptree bias_row;
			for (int j = 0; j < conv2d->get_bias()->rows(); j++)
			{
				boost::property_tree::ptree bias;
				bias.put_value((*conv2d->get_bias())(j));
				bias_row.push_back(std::make_pair("", bias));
				++num_params;
			}
			bias_matrix.push_back(std::make_pair("0", bias_row));
			root.add_child("model.layer." + std::to_string(idx) + ".bias", bias_matrix);
		}


		/*
		* Stores conv2d layer.
		*/
		void store_conv2d_cc(boost::property_tree::ptree& root, const std::shared_ptr<Layer>& current_layer, const size_t idx, size_t& num_params)
		{
			std::shared_ptr<Conv2dConcatLayer> conv2d = std::dynamic_pointer_cast<Conv2dConcatLayer>(current_layer);
			root.put("model.layer." + std::to_string(idx) + ".type", std::to_string(conv2d->get_layer_type()));
			root.put("model.layer." + std::to_string(idx) + ".id", std::to_string(conv2d->get_layer_id()));
			root.put("model.layer." + std::to_string(idx) + ".filter_count", std::to_string(conv2d->get_weights()->rows()));
			ConvolutionParams  conv_params = conv2d->get_conv_params();
			root.put("model.layer." + std::to_string(idx) + ".filter_dim", std::to_string(conv_params.kernel_width()));
			root.put("model.layer." + std::to_string(idx) + ".stride", std::to_string(conv_params.stride()));
			root.put("model.layer." + std::to_string(idx) + ".activation", std::to_string(conv2d->get_activation()->get_type()));
			root.put("model.layer." + std::to_string(idx) + ".crop_layer_id", std::to_string(conv2d->get_crop_layer()->get_layer_id()));
			boost::property_tree::ptree weight_matrix;
			for (int j = 0; j < conv2d->get_weights()->rows(); j++)
			{
				boost::property_tree::ptree weight_row;
				for (int k = 0; k < conv2d->get_weights()->cols(); k++)
				{
					boost::property_tree::ptree weight;
					weight.put_value((*conv2d->get_weights())(j, k));
					weight_row.push_back(std::make_pair("", weight));
					++num_params;
				}
				weight_matrix.push_back(std::make_pair(std::to_string(j), weight_row));
			}
			root.add_child("model.layer." + std::to_string(idx) + ".weights", weight_matrix);
			boost::property_tree::ptree bias_matrix;
			boost::property_tree::ptree bias_row;
			for (int j = 0; j < conv2d->get_bias()->rows(); j++)
			{
				boost::property_tree::ptree bias;
				bias.put_value((*conv2d->get_bias())(j));
				bias_row.push_back(std::make_pair("", bias));
				++num_params;
			}
			bias_matrix.push_back(std::make_pair("0", bias_row));
			root.add_child("model.layer." + std::to_string(idx) + ".bias", bias_matrix);
		}

		/*
		* Stores conv2d transposed layer.
		*/
		void store_conv2d_t(boost::property_tree::ptree& root, const std::shared_ptr<Layer>& current_layer, const size_t idx, size_t& num_params)
		{
			std::shared_ptr<Conv2dTransposeLayer> conv2d = std::dynamic_pointer_cast<Conv2dTransposeLayer>(current_layer);
			root.put("model.layer." + std::to_string(idx) + ".type", std::to_string(conv2d->get_layer_type()));
			root.put("model.layer." + std::to_string(idx) + ".id", std::to_string(conv2d->get_layer_id()));
			root.put("model.layer." + std::to_string(idx) + ".filter_count", std::to_string(conv2d->get_weights()->rows()));
			ConvolutionParams  conv_params = conv2d->get_conv_params();
			root.put("model.layer." + std::to_string(idx) + ".filter_dim", std::to_string(conv_params.kernel_width()));
			root.put("model.layer." + std::to_string(idx) + ".stride", std::to_string(conv_params.stride()));
			root.put("model.layer." + std::to_string(idx) + ".activation", std::to_string(conv2d->get_activation()->get_type()));
			boost::property_tree::ptree weight_matrix;
			for (int j = 0; j < conv2d->get_weights()->rows(); j++)
			{
				boost::property_tree::ptree weight_row;
				for (int k = 0; k < conv2d->get_weights()->cols(); k++)
				{
					boost::property_tree::ptree weight;
					weight.put_value((*conv2d->get_weights())(j, k));
					weight_row.push_back(std::make_pair("", weight));
					++num_params;
				}
				weight_matrix.push_back(std::make_pair(std::to_string(j), weight_row));
			}
			root.add_child("model.layer." + std::to_string(idx) + ".weights", weight_matrix);
			boost::property_tree::ptree bias_matrix;
			boost::property_tree::ptree bias_row;
			for (int j = 0; j < conv2d->get_bias()->rows(); j++)
			{
				boost::property_tree::ptree bias;
				bias.put_value((*conv2d->get_bias())(j));
				bias_row.push_back(std::make_pair("", bias));
				++num_params;
			}
			bias_matrix.push_back(std::make_pair("0", bias_row));
			root.add_child("model.layer." + std::to_string(idx) + ".bias", bias_matrix);
		}

		/*
		* Stores fully-connected layer.
		*/
		void store_fc_layer(boost::property_tree::ptree& root, std::shared_ptr<Layer>& current_layer, size_t idx, size_t& num_params)
		{
			std::shared_ptr<FcLayer> fc = std::dynamic_pointer_cast<FcLayer>(current_layer);
			root.put("model.layer." + std::to_string(idx) + ".type", std::to_string(fc->get_layer_type()));
			root.put("model.layer." + std::to_string(idx) + ".id", std::to_string(fc->get_layer_id()));
			root.put("model.layer." + std::to_string(idx) + ".in_size", std::to_string(fc->get_weights()->cols()));
			root.put("model.layer." + std::to_string(idx) + ".out_size", std::to_string(fc->get_weights()->rows()));
			root.put("model.layer." + std::to_string(idx) + ".activation", std::to_string(fc->get_activation()->get_type()));
			boost::property_tree::ptree kernels;
			size_t row_idx = 0;

			for (int j = 0; j < fc->get_weights()->rows(); j++)
			{
				boost::property_tree::ptree weight_row;
				for (int k = 0; k < fc->get_weights()->cols(); k++)
				{
					boost::property_tree::ptree weight;
					weight.put_value((*fc->get_weights())(j, k));
					weight_row.push_back(std::make_pair("", weight));
					++num_params;
				}
				kernels.push_back(std::make_pair(std::to_string(row_idx++), weight_row));
			}
			root.add_child("model.layer." + std::to_string(idx) + ".weights", kernels);
			boost::property_tree::ptree bias_matrix;
			boost::property_tree::ptree bias_row;
			for (int j = 0; j < fc->get_bias()->rows(); j++)
			{
				boost::property_tree::ptree bias;
				bias.put_value((*fc->get_bias())(j));
				bias_row.push_back(std::make_pair("", bias));
				++num_params;
			}
			bias_matrix.push_back(std::make_pair("0", bias_row));
			root.add_child("model.layer." + std::to_string(idx) + ".bias", bias_matrix);
		}


		/*
		* Stores input layer.
		*/
		void store_input(boost::property_tree::ptree& root, std::shared_ptr<Layer>& current_layer, size_t idx)
		{
			std::shared_ptr<InputLayer> layer = std::dynamic_pointer_cast<InputLayer>(current_layer);
			root.put("model.layer." + std::to_string(idx) + ".type", std::to_string(layer->get_layer_type()));
			root.put("model.layer." + std::to_string(idx) + ".id", std::to_string(layer->get_layer_id()));
			root.put("model.layer." + std::to_string(idx) + ".in_size", std::to_string(layer->get_output()->cols()));
			root.put("model.layer." + std::to_string(idx) + ".in_channels", std::to_string(layer->get_output()->rows()));
		}

		/*
		* Stores maxpool2d layer.
		*/
		void store_maxpool2d(boost::property_tree::ptree& root, std::shared_ptr<Layer>& current_layer, size_t idx)
		{
			std::shared_ptr<MaxPool2dLayer> layer = std::dynamic_pointer_cast<MaxPool2dLayer>(current_layer);
			root.put("model.layer." + std::to_string(idx) + ".type", std::to_string(layer->get_layer_type()));
			root.put("model.layer." + std::to_string(idx) + ".id", std::to_string(layer->get_layer_id()));
			root.put("model.layer." + std::to_string(idx) + ".window_size", std::to_string(layer->get_window_size()));
			root.put("model.layer." + std::to_string(idx) + ".stride", std::to_string(layer->get_stride()));
		}

		/*
		* Stores layer depending on type.
		*/
		void store(Graph& graph, size_t epoch = 0, size_t batch = 0, double valid_acc = 0) override
		{
			size_t num_params = 0;
			boost::property_tree::ptree root;
			size_t num_layers = graph.get_num_layers();
			for (size_t layer = 0; layer < num_layers; ++layer)
			{
				store_layer(root, graph.get_layer_by_idx(layer), layer, num_params);
			}
			root.put("model.meta.num_params", std::to_string(num_params));
			root.put("model.meta.num_layers", std::to_string(num_layers));
			root.put("model.meta.num_epoch", std::to_string(epoch));
			root.put("model.meta.num_batch", std::to_string(batch));
			if (valid_acc > 0.f)
			{
				root.put("model.meta.valid_accuracy", std::to_string(valid_acc));
			}
			boost::property_tree::write_json(this->_model_path, root);
		}

		std::shared_ptr<Layer> restore_fc(boost::property_tree::ptree& root, std::shared_ptr<Layer> previous_layer, size_t layer_idx)
		{
			const size_t out_size = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".out_size"));
			const std::string activation_string = root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".activation");
			const ActivationType activation_type = (ActivationType)std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".activation"));
			std::shared_ptr<Layer> current_layer = std::shared_ptr<FcLayer>(new FcLayer(previous_layer, out_size, get_activation_ptr(activation_type)));
			for (size_t j = 0; j < out_size; ++j)
			{
				size_t k = 0;
				for (boost::property_tree::ptree::value_type &weight : root.get_child("model.layer." + std::to_string(layer_idx) + ".weights." + std::to_string(j)))
				{
					(*current_layer->get_weights())(j, k) = std::stof(weight.second.data());
					++k;
				}
			}
			size_t k = 0;
			for (boost::property_tree::ptree::value_type &bias : root.get_child("model.layer." + std::to_string(layer_idx) + ".bias.0" ))
			{
				(*current_layer->get_bias())(k) = std::stof(bias.second.data());
				++k;
			}
			const size_t layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".id"));
			current_layer->set_layer_id(layer_id);
			return current_layer;
		}
		/*
		Restores con2vd layer
		*/
		std::shared_ptr<Layer> restore_conv2d(boost::property_tree::ptree& root, std::shared_ptr<Layer> previous_layer, size_t layer_idx)
		{
			size_t filter_count = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".filter_count"));
			size_t filter_dim = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".filter_dim"));
			size_t stride = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".stride"));
			const size_t pad = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".pad"));
			ActivationType activation_type = (ActivationType)std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".activation"));
			std::shared_ptr<Layer> current_layer = Conv2dLayer::create(previous_layer, filter_count, filter_dim, stride, (PaddingPolicy) pad, activation_type);
			for (size_t j = 0; j < filter_count; ++j)
			{
				size_t k = 0;
				for (boost::property_tree::ptree::value_type &weight : root.get_child("model.layer." + std::to_string(layer_idx) + ".weights." + std::to_string(j)))
				{
					(*current_layer->get_weights())(j, k++) = std::stof(weight.second.data());
				}
			}
			size_t k = 0;
			for (boost::property_tree::ptree::value_type &bias : root.get_child("model.layer." + std::to_string(layer_idx) + ".bias.0"))
			{
				(*current_layer->get_bias())(k) = std::stof(bias.second.data());
				++k;
			}
			const size_t layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".id"));
			current_layer->set_layer_id(layer_id);
			return current_layer;
		}

		/*
		Restores conv2d_cc layer
		*/
		std::shared_ptr<Layer> restore_conv2d_cc(boost::property_tree::ptree& root, std::shared_ptr<Layer> previous_layer, size_t layer_idx)
		{
			size_t filter_count = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".filter_count"));
			size_t filter_dim = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".filter_dim"));
			size_t stride = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".stride"));
			ActivationType activation_type = (ActivationType)std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".activation"));
			const size_t crop_layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".crop_layer_id"));
			std::shared_ptr<Layer> crop_layer = get_layer_ptr(crop_layer_id);
			std::shared_ptr<Layer> current_layer = std::make_shared<Conv2dConcatLayer>(crop_layer, previous_layer, filter_count, filter_dim, stride, get_activation_ptr(activation_type));
			for (size_t j = 0; j < filter_count; ++j)
			{
				size_t k = 0;
				for (boost::property_tree::ptree::value_type &weight : root.get_child("model.layer." + std::to_string(layer_idx) + ".weights." + std::to_string(j)))
				{
					(*current_layer->get_weights())(j, k++) = std::stof(weight.second.data());
				}
			}
			size_t k = 0;
			for (boost::property_tree::ptree::value_type &bias : root.get_child("model.layer." + std::to_string(layer_idx) + ".bias.0"))
			{
				(*current_layer->get_bias())(k) = std::stof(bias.second.data());
				++k;
			}
			const size_t layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".id"));
			current_layer->set_layer_id(layer_id);
			return current_layer;
		}

		/*
		Restores conv2d_t layer
		*/
		std::shared_ptr<Layer> restore_conv2d_t(boost::property_tree::ptree& root, std::shared_ptr<Layer> previous_layer, size_t layer_idx)
		{
			size_t filter_count = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".filter_count"));
			size_t filter_dim = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".filter_dim"));
			size_t stride = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".stride"));
			ActivationType activation_type = (ActivationType)std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".activation"));
			std::shared_ptr<Layer> current_layer = std::make_shared<Conv2dTransposeLayer>(previous_layer, filter_count, filter_dim, stride, get_activation_ptr(activation_type));
			for (size_t j = 0; j < filter_count; ++j)
			{
				size_t k = 0;
				for (boost::property_tree::ptree::value_type &weight : root.get_child("model.layer." + std::to_string(layer_idx) + ".weights." + std::to_string(j)))
				{
					(*current_layer->get_weights())(j, k++) = std::stof(weight.second.data());
				}
			}
			size_t k = 0;
			for (boost::property_tree::ptree::value_type &bias : root.get_child("model.layer." + std::to_string(layer_idx) + ".bias.0"))
			{
				(*current_layer->get_bias())(k) = std::stof(bias.second.data());
				++k;
			}
			const size_t layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".id"));
			current_layer->set_layer_id(layer_id);
			return current_layer;
		}

		std::shared_ptr<Layer> restore_input(boost::property_tree::ptree& root, size_t layer_idx)
		{
			size_t in_size = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".in_size"));
			size_t in_channels = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".in_channels"));
			std::shared_ptr<Layer> current_layer = std::make_shared<InputLayer>((size_t)sqrt(in_size), in_channels);
			const size_t layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".id"));
			current_layer->set_layer_id(layer_id);
			return current_layer;
		}

		/*
		* Method restores Maxpool2D layer
		*/
		std::shared_ptr<Layer> restore_maxpool2d(boost::property_tree::ptree& root, std::shared_ptr<Layer> previous_layer, size_t layer_idx)
		{
			size_t window_size = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".window_size"));
			size_t stride = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".stride"));
			std::shared_ptr<Layer> current_layer = std::make_shared<MaxPool2dLayer>(previous_layer, window_size, stride);
			const size_t layer_id = std::stoi(root.get<std::string>("model.layer." + std::to_string(layer_idx) + ".id"));
			current_layer->set_layer_id(layer_id);
			return current_layer;
		}

		/*
		* Method restores Graph instance from a json file.
		*/
		void restore(Graph* graph) override
		{
			std::cout << "INFO:: (Saver) Restoring model from " << _model_path << "." << std::endl;
			boost::property_tree::ptree root;
			boost::property_tree::read_json(this->_model_path, root);
			size_t num_layers = std::stoi(root.get<std::string>("model.meta.num_layers"));
			std::shared_ptr<Layer> previous_layer = nullptr;
			for (size_t i = 0; i < num_layers; ++i)
			{
				LayerType layer_type = (LayerType) ( std::stoi(root.get<std::string>("model.layer." + std::to_string(i) + ".type")));
				if (layer_type == CONV2D_LAYER)
				{
					previous_layer = restore_conv2d(root, previous_layer, i);
					this->_layer_map[previous_layer->get_layer_id()] = previous_layer;
				}
				else if (layer_type == CONV2D_T_LAYER)
				{
					previous_layer = restore_conv2d_t(root, previous_layer, i);
					this->_layer_map[previous_layer->get_layer_id()] = previous_layer;
				}
				else if (layer_type == INPUT_LAYER)
				{
					previous_layer = restore_input(root, i);
					this->_layer_map[previous_layer->get_layer_id()] = previous_layer;
				}
				else  if (layer_type == FC_LAYER)
				{
					previous_layer = restore_fc(root, previous_layer, i);
					this->_layer_map[previous_layer->get_layer_id()] = previous_layer;
				}
				else  if (layer_type == MAXPOOL2D_LAYER)
				{
					previous_layer = restore_maxpool2d(root, previous_layer, i);
					this->_layer_map[previous_layer->get_layer_id()] = previous_layer;
				}
				else  if (layer_type == CONV2D_CC_LAYER)
				{
					previous_layer = restore_conv2d_cc(root, previous_layer, i);
					this->_layer_map[previous_layer->get_layer_id()] = previous_layer;
				}
			}
			//Wrap in graph instance
			graph->from_ptr(previous_layer);
			std::cout << "INFO:: (Saver) Successfully restored model from " << _model_path << "." << std::endl;
		}


	};

}

#endif
