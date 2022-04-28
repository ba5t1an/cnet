/**
Common functions and typedefs.

@file cnet_conf.hpp
@author Bastian 
*/

#ifndef CNET_CONF_HPP
#define CNET_CONF_HPP

#include <string>
#include <boost/assign/list_of.hpp>
#include <boost/algorithm/string.hpp>
#include "cnet_layer.hpp"
#include "cnet_saver.hpp"
#include "cnet_graph.hpp"
#include "cnet_estimator.hpp"
#include "cnet_solver2.hpp"

namespace Cnet
{

    enum TaskType
    {
        TASK_CLASSIFICATION = 0, TASK_SEGMENTATION = 1
    };

    struct TrainParams
    {
        size_t batch_size;
        size_t max_iter;
        size_t test_iter;
        size_t display_iter;
        size_t save_iter;
        std::string save_path;
        size_t num_threads;
        bool resume_training;
        std::string model_path;
        bool print_model;
    };

    struct SetParams
    {
        std::string data_file;
        size_t num_items;
        size_t num_threads;
        bool shuffle;
    };


	struct PreprocParams
	{
		size_t resize_to;
		size_t pad_to;
		size_t crop_to;
	};

    struct DataParams
    {
        bool encode_one_hot;
        size_t num_classes;
        SetParams val;
        SetParams train;
		bool input_preproc_enabled;
		bool label_preproc_enabled;
		PreprocParams preproc_input;
		PreprocParams preproc_label;
    };

    struct ImageLoggingParams
    {
        bool logging_enabled;
        bool log_input;
        bool log_output;
		bool use_argmax;
        bool log_label;
        size_t log_iter;
        std::string log_dir;
        size_t log_num;
    };



    class Configuration
    {

    private:
        boost::property_tree::ptree _root;
        std::string _cfg_path;

        std::map<size_t, std::shared_ptr<Layer> > _layer_map;

        const std::map<std::string, InitializerType> _initializer_keys = {{"he_normal", HE_NORMAL},
                                                                          {"he_uniform", HE_UNIFORM}};

		const std::map<std::string, ActivationType> _activation_keys = { {"relu", RELU},
																		 {"leaky_relu", LEAKY_RELU},
																		 {"softmax", SOFTMAX } };

        const std::map<std::string, TaskType> _task_keys = {{"classification",TASK_CLASSIFICATION},
                                                            {"segmentation", TASK_SEGMENTATION}};

        std::shared_ptr<Layer> parse_input_layer(boost::property_tree::ptree::value_type &layer)
        {
            const size_t input_width = layer.second.get<size_t>("input_width");
            const size_t num_channels = layer.second.get<size_t>("input_channels");
            const size_t idx = layer.second.get<size_t>("id");
            std::shared_ptr<Layer> current_layer = InputLayer::create(input_width, num_channels);
            current_layer->set_layer_id(idx);
            _layer_map[idx] = current_layer;
            return current_layer;
        }

        std::shared_ptr<Layer>
        parse_conv2d_layer(std::shared_ptr<Layer>& prev_layer, boost::property_tree::ptree::value_type &layer)
        {
            const size_t num_kernels = layer.second.get<size_t>("num_kernels");
            const size_t kernel_width = layer.second.get<size_t>("kernel_width");
            const size_t stride = layer.second.get<size_t>("stride");
            const std::string padding = layer.second.get<std::string>("padding");
            PaddingPolicy p = VALID;
            if (boost::algorithm::to_lower_copy(padding).compare("same") == 0) {
                p = SAME;
            }
            const std::string activation = layer.second.get<std::string>("activation");
            const std::string initializer = layer.second.get<std::string>("initializer");
            const size_t idx = layer.second.get<size_t>("id");
            ActivationType act = _activation_keys.find(boost::algorithm::to_lower_copy(activation))->second;
            InitializerType init = _initializer_keys.find(boost::algorithm::to_lower_copy(initializer))->second;
            std::shared_ptr<Layer> current_layer = Conv2dLayer::create(prev_layer, num_kernels, kernel_width, stride, p,
                                                                       act, init);
            current_layer->set_layer_id(idx);
            _layer_map[idx] = current_layer;
            return current_layer;
        }

        std::shared_ptr<Layer>
        parse_dropout_layer(std::shared_ptr<Layer> &prev_layer, boost::property_tree::ptree::value_type &layer)
        {
            const float prob = layer.second.get<float>("dropout_prob");
            const size_t idx = layer.second.get<size_t>("id");
            std::shared_ptr<Layer> current_layer = DropoutLayer::create(prev_layer, prob);
            current_layer->set_layer_id(idx);
            _layer_map[idx] = current_layer;
            return current_layer;
        }

        std::shared_ptr<Layer>
        parse_conv2d_t_layer(std::shared_ptr<Layer> &prev_layer, boost::property_tree::ptree::value_type &layer)
        {
            const size_t num_kernels = layer.second.get<size_t>("num_kernels");
            const size_t kernel_width = layer.second.get<size_t>("kernel_width");
            const size_t stride = layer.second.get<size_t>("stride");
            const std::string activation = layer.second.get<std::string>("activation");
            const std::string initializer = layer.second.get<std::string>("initializer");
            const size_t idx = layer.second.get<size_t>("id");
            ActivationType act = _activation_keys.find(boost::algorithm::to_lower_copy(activation))->second;
            InitializerType init = _initializer_keys.find(boost::algorithm::to_lower_copy(initializer))->second;
            std::shared_ptr<Layer> current_layer = Conv2dTransposeLayer::create(prev_layer, num_kernels, kernel_width,
                                                                                stride, act, init);
            current_layer->set_layer_id(idx);
            _layer_map[idx] = current_layer;
            return current_layer;
        }

        std::shared_ptr<Layer>
        parse_conv2d_concat_layer(std::shared_ptr<Layer> &prev_layer, boost::property_tree::ptree::value_type &layer)
        {
            const size_t concat_layer_id = layer.second.get<size_t>("concat_layer_id");
            std::shared_ptr<Layer> concat_layer = _layer_map.find(concat_layer_id)->second;
            const size_t num_kernels = layer.second.get<size_t>("num_kernels");
            const size_t kernel_width = layer.second.get<size_t>("kernel_width");
            const size_t stride = layer.second.get<size_t>("stride");
            const std::string activation = layer.second.get<std::string>("activation");
            const std::string initializer = layer.second.get<std::string>("initializer");
            const size_t idx = layer.second.get<size_t>("id");
            ActivationType act = _activation_keys.find(boost::algorithm::to_lower_copy(activation))->second;
            InitializerType init = _initializer_keys.find(boost::algorithm::to_lower_copy(initializer))->second;
            std::shared_ptr<Layer> current_layer = Conv2dConcatLayer::create(prev_layer, concat_layer, num_kernels,
                                                                             kernel_width, stride, act, init);
            concat_layer->set_bwd_skip(current_layer);
            current_layer->set_layer_id(idx);
            _layer_map[idx] = current_layer;
            return current_layer;
        }

        std::shared_ptr<Layer>
        parse_maxpool2d_layer(std::shared_ptr<Layer> &prev_layer, boost::property_tree::ptree::value_type &layer)
        {
            const size_t window_width = layer.second.get<size_t>("window_width");
            const size_t stride = layer.second.get<size_t>("stride");
            const size_t idx = layer.second.get<size_t>("id");
            std::shared_ptr<Layer> current_layer = MaxPool2dLayer::create(prev_layer, window_width, stride);
            current_layer->set_layer_id(idx);
            _layer_map[idx] = current_layer;
            return current_layer;
        }

        std::shared_ptr<Layer>
        parse_fc_layer(std::shared_ptr<Layer> &prev_layer, boost::property_tree::ptree::value_type &layer)
        {
            const size_t num_outputs = layer.second.get<size_t>("num_outputs");
            const std::string activation = layer.second.get<std::string>("activation");
            const std::string initializer = layer.second.get<std::string>("initializer");
            const size_t idx = layer.second.get<size_t>("id");
            ActivationType act = _activation_keys.find(boost::algorithm::to_lower_copy(activation))->second;
            InitializerType init = _initializer_keys.find(boost::algorithm::to_lower_copy(initializer))->second;
            std::shared_ptr<Layer> current_layer = FcLayer::create(prev_layer, num_outputs, act, init);
            current_layer->set_layer_id(idx);
            return current_layer;
        }

        std::shared_ptr<Layer> iterate_layers(boost::property_tree::ptree &root)
        {
            std::shared_ptr<Layer> previous;
            for (boost::property_tree::ptree::value_type &layer : root.get_child("model.layer")) {

                if (layer.second.get<std::string>("type").compare("Input") == 0) {
                    previous = parse_input_layer(layer);
                } else if (layer.second.get<std::string>("type").compare("Conv2d") == 0) {
                    previous = parse_conv2d_layer(previous, layer);
                } else if (layer.second.get<std::string>("type").compare("MaxPool2d") == 0) {
                    previous = parse_maxpool2d_layer(previous, layer);
                } else if (layer.second.get<std::string>("type").compare("Conv2dConcat") == 0) {
                    previous = parse_conv2d_concat_layer(previous, layer);
                } else if (layer.second.get<std::string>("type").compare("Fc") == 0) {
                    previous = parse_fc_layer(previous, layer);
                } else if (layer.second.get<std::string>("type").compare("Dropout") == 0) {
                    previous = parse_dropout_layer(previous, layer);
                } else if (layer.second.get<std::string>("type").compare("Conv2dTranspose") == 0) {
                    previous = parse_conv2d_t_layer(previous, layer);
                }
            }
            return previous;
        }

        std::unique_ptr<Solver> parse_adam(boost::property_tree::ptree &root)
        {
            const float lr = root.get<float>("estimator.solver.lr");
            const float beta1 = root.get<float>("estimator.solver.beta1");
            const float beta2 = root.get<float>("estimator.solver.beta2");
            const float epsilon = root.get<float>("estimator.solver.epsilon");
            return std::unique_ptr<AdamSolver>(new AdamSolver(lr, beta1, beta2, epsilon));
        }

        std::unique_ptr<Solver> parse_sgd(boost::property_tree::ptree &root)
        {
            const float lr = root.get<float>("estimator.solver.lr");
            const float momentum = root.get<float>("estimator.solver.momentum");
            return std::unique_ptr<SgdSolver>(new SgdSolver(lr, momentum));
        }

        std::unique_ptr<Solver> parse_solver(boost::property_tree::ptree &root)
        {
            std::string type = root.get<std::string>("estimator.solver.type");
            if (type.compare("Sgd") == 0) {
                return parse_sgd(root);
            }
            return parse_adam(root);
        }

        std::unique_ptr<Loss> parse_loss(boost::property_tree::ptree &root)
        {
            std::string type = root.get<std::string>("estimator.loss.type");
            if (type.compare("CrossEntropy") == 0) {
                return std::unique_ptr<CrossEntropyLoss>(new CrossEntropyLoss());
            } else if (type.compare("BinaryCrossEntropy") == 0) {
                return std::unique_ptr<BinaryCrossEntropyLoss>(new BinaryCrossEntropyLoss());
            } else if (type.compare("SparseSoftmaxCrossEntropy") == 0) {
                return std::unique_ptr<SparseSoftmaxCrossEntropyLoss>(new SparseSoftmaxCrossEntropyLoss());
            }
            return std::unique_ptr<MseLoss>(new MseLoss());
        }


        void parse_dataset_params(boost::property_tree::ptree &root, SetParams &params, std::string type)
        {
            params.num_threads = 1;
            boost::optional<boost::property_tree::ptree &> child = root.get_child_optional(
                    "data." + type + ".num_threads");
            if (child) {
                params.num_threads = root.get<size_t>("data." + type + ".num_threads");
            }
            params.shuffle = false;
            child = root.get_child_optional("data." + type + ".shuffle");
            if (child) {
                int shuffle = root.get<int>("data." + type + ".shuffle");
                params.shuffle = (shuffle == 0 ? false : true);
            }
            params.num_items = 0;
            child = root.get_child_optional("data." + type + ".num_items");
            if (child) {
                params.num_items = root.get<size_t>("data." + type + ".num_items");
            }
            params.data_file = root.get<std::string>("data." + type + ".data_file");
        }

        void parse_data_params(boost::property_tree::ptree &root, DataParams &params)
        {
            params.encode_one_hot = false;
            boost::optional<boost::property_tree::ptree &> child = root.get_child_optional("data.encode_one_hot");
            if (child) {
                int encode_one_hot = root.get<int>("data.encode_one_hot");
                params.encode_one_hot = (encode_one_hot == 0 ? false : true);
            }
            params.num_classes = 0;
            child = root.get_child_optional("data.num_classes");
            if (child) {
                params.num_classes = root.get<size_t>("data.num_classes");
            }
            SetParams val;
            parse_dataset_params(root, val, "val");
            SetParams train;
            parse_dataset_params(root, train, "train");
            params.val = val;
            params.train = train;
        }

        void parse_image_logging(boost::property_tree::ptree &root, ImageLoggingParams &params)
        {
            params.logging_enabled = false;
            boost::optional<boost::property_tree::ptree &> child = root.get_child_optional("estimator.image_logging");
            if (child) {
                int logging_enabled = root.get<int>("estimator.image_logging.logging_enabled");
                params.logging_enabled = (logging_enabled == 0 ? false : true);
                if (params.logging_enabled) {
                    int log_input = root.get<int>("estimator.image_logging.log_input");
                    params.log_input = (log_input == 0 ? false : true);
					child = root.get_child_optional("estimator.image_logging.use_argmax");
					if (child)
					{
						int use_argmax = root.get<int>("estimator.image_logging.use_argmax");
						params.use_argmax = (use_argmax == 0 ? false : true);
						
					}
					int log_output = root.get<int>("estimator.image_logging.log_output");
					params.log_output = (log_output == 0 ? false : true);
						
                    int log_label = root.get<int>("estimator.image_logging.log_label");
                    params.log_label = (log_label == 0 ? false : true);

                    params.log_iter = root.get<size_t>("estimator.image_logging.log_iter");
                    params.log_dir = root.get<std::string>("estimator.image_logging.log_dir");
                    params.log_num = root.get<size_t>("estimator.image_logging.log_num");
                }
            }

        }

        void parse_train_params(boost::property_tree::ptree &root, TrainParams &params)
        {
            params.batch_size = root.get<size_t>("estimator.batch_size");
            params.test_iter = root.get<size_t>("estimator.test_iter");
            params.max_iter = root.get<size_t>("estimator.max_iter");
            params.display_iter = root.get<size_t>("estimator.display_iter");
            params.save_iter = root.get<size_t>("estimator.save_iter");
            params.save_path = root.get<std::string>("estimator.save_path");
            params.resume_training = false;
            boost::optional<boost::property_tree::ptree &> child = root.get_child_optional("estimator.resume_training");
            if (child) {
                int resume_training = root.get<int>("estimator.resume_training");
                params.resume_training = (resume_training == 0 ? false : true);
                if (params.resume_training) {
                    params.model_path = root.get<std::string>("estimator.model_path");
                }
            }
            params.num_threads = 0;
            child = root.get_child_optional("estimator.num_threads");
            if (child) {
                params.num_threads = root.get<size_t>("estimator.num_threads");
            }
            params.print_model = false;
            child = root.get_child_optional("estimator.debug.print_model");
            if (child) {
                int print_model = root.get<int>("estimator.debug.print_model");
                params.print_model = (print_model == 0 ? false : true);
            }
        }

		void parse_preproc(boost::property_tree::ptree &root, PreprocParams& params, std::string type)
		{
			params.resize_to = 0;
			params.pad_to = 0;
			params.crop_to = 0;
			boost::optional<boost::property_tree::ptree &> child = root.get_child_optional("data.preproc." + type + ".resize_to");
			if (child)
			{
				params.resize_to = root.get<int>("data.preproc." + type + ".resize_to");
			}
			child = root.get_child_optional("data.preproc." + type + ".pad_to");
			if (child)
			{
				params.pad_to = root.get<int>("data.preproc." + type + ".pad_to");
			}
			child = root.get_child_optional("data.preproc." + type + ".crop_to");
			if (child)
			{
				params.crop_to = root.get<int>("data.preproc." + type + ".crop_to");
			}
		}

		void parse_preproc_params(boost::property_tree::ptree &root, DataParams& params)
		{
			params.label_preproc_enabled = false;
			params.input_preproc_enabled = false;
			boost::optional<boost::property_tree::ptree &> child = root.get_child_optional("data.preproc");
			if (child)
			{
				child = root.get_child_optional("data.preproc.input");
				if (child)
				{
					params.input_preproc_enabled = true;
					PreprocParams preproc_params;
					parse_preproc(root, preproc_params, "input");
					params.preproc_input = preproc_params;
				}
				child = _root.get_child_optional("data.preproc.label");
				if (child)
				{
					params.label_preproc_enabled = true;
					PreprocParams preproc_params;
					parse_preproc(root, preproc_params, "label");
					params.preproc_label = preproc_params;
				}
			}
		}


    public:

        Configuration(const std::string &cfg_path)
        {
            _cfg_path = cfg_path;
            boost::property_tree::read_json(_cfg_path, _root);
        }

        void parse_model(Graph &graph)
        {
            std::shared_ptr<Layer> layer = iterate_layers(_root);
            graph.from_ptr(layer);
        }

        std::unique_ptr<Solver> parse_solver()
        {
            return std::move(parse_solver(_root));
        }

        std::unique_ptr<Loss> parse_loss()
        {
            return std::move(parse_loss(_root));
        }

        void parse_train_params(TrainParams &params)
        {
            parse_train_params(_root, params);
        }

        void parse_data_params(DataParams &params)
        {
            parse_data_params(_root, params);
			parse_preproc_params(_root, params);
        }

        TaskType parse_task_type()
        {
            std::string task_type = _root.get<std::string>("task.type");
            return _task_keys.find(boost::algorithm::to_lower_copy(task_type))->second;
        }

        
		void parse_image_logging(ImageLoggingParams &params)
		{
			parse_image_logging(_root, params);
		}
    };

}

#endif //CNET_CONF_HPP
