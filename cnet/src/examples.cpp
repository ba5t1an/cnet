
//To use MKL for certrain BLAS operations

//#define EIGEN_USE_MKL_ALL

#include <thread>
#include <chrono>

#include <iostream>

#include "cnet_common.hpp"
#include "cnet_graph.hpp"
#include "cnet_saver.hpp"
#include "cnet_parser.hpp"
#include "cnet_dataset.hpp"
#include "cnet_preproc.hpp"
#include "cnet_estimator.hpp"
#include "cnet_layer.hpp"
#include "cnet_evaluator.hpp"
#include "cnet_conf.hpp"


void add_preproc_operations(std::shared_ptr<Cnet::Pipeline>& pipeline, Cnet::PreprocParams& params, std::string type)
{
	if (params.resize_to > 0)
	{
		std::cout << "INFO::" << "(Main) (" << type << ") Adding preprocessing operation: Resize = " << params.resize_to << "x" << params.resize_to << std::endl;
		pipeline->add_operation(std::unique_ptr<Cnet::ResizeOp>(new Cnet::ResizeOp(params.resize_to)));
	}
	if (params.pad_to > 0)
	{
		std::cout << "INFO::" << "(Main) (" << type << ") Adding preprocessing operation: Pad = " << params.pad_to << "x" << params.pad_to << std::endl;
		pipeline->add_operation(std::unique_ptr<Cnet::PadOp>(new Cnet::PadOp(params.pad_to)));
	}
	if (params.crop_to > 0)
	{
		std::cout << "INFO::" << "(Main) (" << type << ") Adding preprocessing operation: Crop = " << params.crop_to << "x" << params.crop_to << std::endl;
		pipeline->add_operation(std::unique_ptr<Cnet::CropOp>(new Cnet::CropOp(params.crop_to)));
	}
}

void run_classification_task(Cnet::Graph& graph, Cnet::TrainParams& params, Cnet::DataParams& data_params, Cnet::ImageLoggingParams& logging_params, std::unique_ptr<Cnet::Loss> loss, std::unique_ptr<Cnet::Solver> solver)
{
	Cnet::DatasetParams dparams;
	dparams.encode_one_hot = data_params.encode_one_hot;
	dparams.num_classes = data_params.num_classes;
	std::shared_ptr<Cnet::Pipeline> input_pipeline = std::make_shared<Cnet::Pipeline>();
	input_pipeline->add_operation(std::unique_ptr<Cnet::ScaleOp>(new Cnet::ScaleOp(255.f)));
	if (data_params.input_preproc_enabled)
	{
		add_preproc_operations(input_pipeline, data_params.preproc_input, "IN");
	}
	
	int ret = 0;
	Cnet::OnlineDataset train_data(dparams, input_pipeline, data_params.train.num_threads);
	if (params.max_iter > 0)
	{
		train_data.enabled_auto_repeat();
		std::cout << "INFO::" << "(Main) Loading training data from " << data_params.train.data_file << std::endl;
		int ret = Cnet::read_caffe_format(train_data, data_params.train.data_file, data_params.train.num_items);
		if (ret == 0)
		{
			std::cout << "ERROR::" << "(Main) Unable to load train data from " << data_params.train.data_file << std::endl;
			return;
		}
	}
	
	Cnet::OnlineDataset val_data(dparams, input_pipeline, data_params.val.num_threads);
	std::cout << "INFO::" << "(Main) Loading validation data from " << data_params.val.data_file << std::endl;
	ret = Cnet::read_caffe_format(val_data, data_params.val.data_file, data_params.val.num_items);
	if (ret == 0)
	{
		std::cout << "ERROR::" << "(Main) Unable to load validation data from " << data_params.val.data_file << std::endl;
		return;
	}

	std::shared_ptr<Cnet::Evaluator> evaluator = std::make_shared<Cnet::ClassificationEvaluator>();
	//run estimator
	if (params.max_iter > 0)
	{
		std::cout << "INFO::" << "(Main) Running estimator..." << std::endl;
		Cnet::Estimator estimator(graph, evaluator, std::move(loss), std::move(solver), std::unique_ptr<Cnet::JsonSaver>(new Cnet::JsonSaver(params.save_path)), params.num_threads);
		if (logging_params.logging_enabled)
		{
			estimator.attach_image_logger(logging_params.log_dir, logging_params.log_input, false, false, logging_params.log_iter, logging_params.log_num);
		}
		estimator.train(train_data, val_data, params.batch_size, params.max_iter, params.display_iter, params.save_iter, params.test_iter);
	}
	float score = evaluator->test(graph, val_data);
	std::cout << "INFO::" << "(Main) Test after training scored " << std::to_string(score) << std::endl;
}

void run_segmentation_task(Cnet::Graph& graph, Cnet::TrainParams& params, Cnet::DataParams& data_params, Cnet::ImageLoggingParams& logging_params, std::unique_ptr<Cnet::Loss> loss, std::unique_ptr<Cnet::Solver> solver)
{
	std::shared_ptr<Cnet::Pipeline> input_pipeline = std::make_shared<Cnet::Pipeline>();
	input_pipeline->add_operation(std::unique_ptr<Cnet::ScaleOp>(new Cnet::ScaleOp(255.f)));
	if (data_params.input_preproc_enabled)
	{

		add_preproc_operations(input_pipeline, data_params.preproc_input, "IN");
	}
	std::shared_ptr<Cnet::Pipeline> label_pipeline = std::make_shared<Cnet::Pipeline>();
	label_pipeline->add_operation(std::unique_ptr<Cnet::ScaleOp>(new Cnet::ScaleOp(255.f)));
	if (data_params.label_preproc_enabled)
	{
		add_preproc_operations(label_pipeline, data_params.preproc_label, "LBL");
	}
	Cnet::OnlineDataset train_data(input_pipeline, label_pipeline, data_params.train.num_threads);
	train_data.enabled_auto_repeat();
	Cnet::OnlineDataset val_data(input_pipeline, label_pipeline, data_params.val.num_threads);
	std::cout << "INFO::" << "(Main) Loading training data from " << data_params.train.data_file << std::endl;
	int ret = Cnet::read_seg_format(train_data, data_params.train.data_file, ';', data_params.train.num_items);
	if (ret == 0)
	{
		std::cout << "ERROR::" << "(Main) Unable to load train data from " << data_params.train.data_file << std::endl;
		return;
	}
	std::cout << "INFO::" << "(Main) Loading validation data from " << data_params.val.data_file << std::endl;
	ret = Cnet::read_seg_format(val_data, data_params.val.data_file, ';', data_params.val.num_items);
	if (ret == 0)
	{
		std::cout << "ERROR::" << "(Main) Unable to load validation data from " << data_params.val.data_file << std::endl;
		return;
	}
	//run estimator
	std::cout << "INFO::" << "(Main) Running estimator..." << std::endl;
	Cnet::Estimator estimator(graph, nullptr, std::move(loss), std::move(solver), std::unique_ptr<Cnet::JsonSaver>(new Cnet::JsonSaver(params.save_path)), params.num_threads);
	if (logging_params.logging_enabled)
	{
		estimator.attach_image_logger(logging_params.log_dir, logging_params.log_input, logging_params.log_output, logging_params.log_label, logging_params.log_iter, logging_params.log_num, logging_params.use_argmax);
	}
	estimator.train(train_data, val_data, params.batch_size, params.max_iter, params.display_iter, params.save_iter, params.test_iter);
	std::cout << "INFO::" << "(Main) Training done." << std::endl;
}

int process_config(const std::string& path)
{
	std::cout << "INFO::" << "(Main) Loading configuration file from " << path << std::endl;
	if (!Cnet::file_exists(path))
	{
		std::cout << "INFO::" << "(Main) Unable to load file... " << std::endl;
		return 0;
	}
	Cnet::Configuration conf(path);
	Cnet::TrainParams params;
	conf.parse_train_params(params);
	Cnet::DataParams data_params;
	conf.parse_data_params(data_params);
	Cnet::Graph graph;
	if (params.resume_training)
	{
		std::cout << "INFO::" << "(Main) Resuming training from existing model file " << params.model_path << std::endl;
		Cnet::JsonSaver saver(params.model_path);
		saver.restore(&graph);
	}
	else
	{
		std::cout << "INFO::" << "(Main) Loading model..." << std::endl;
		conf.parse_model(graph);
	}
	if (params.print_model)
	{
		std::cout << "DEBUG::" << "(Main) Printing model:" << std::endl;
		Cnet::ShapeVisitor visitor;
		Cnet::MatrixRm empty_input = Cnet::MatrixRm::Zero(graph.get_input_channels(), graph.get_input_size());
		graph.visit(&empty_input, &visitor);
	}
	Cnet::ImageLoggingParams image_logging_params;
	conf.parse_image_logging(image_logging_params);
	std::unique_ptr<Cnet::Solver> solver = conf.parse_solver();
	std::unique_ptr<Cnet::Loss> loss = conf.parse_loss();
	if (conf.parse_task_type() == Cnet::TASK_CLASSIFICATION)
	{
		std::cout << "INFO::" << "(Main) Running classification task" << std::endl;
		run_classification_task(graph, params, data_params, image_logging_params, std::move(loss), std::move(solver));
	}
	else
	{
		std::cout << "INFO::" << "(Main) Running segmentation task" << std::endl;
		run_segmentation_task(graph, params, data_params, image_logging_params, std::move(loss), std::move(solver));
	}

	return 1;
}


void process_program_options(const int argc, const char *const argv[])
{
	process_config("somepath");
}

void train_mnist()
{
	Cnet::DatasetParams params;
	params.encode_one_hot = true;
	params.num_classes = 10;
	std::shared_ptr<Cnet::Pipeline> pipeline = std::make_shared<Cnet::Pipeline>();
	pipeline->add_operation(std::unique_ptr<Cnet::ScaleOp>(new Cnet::ScaleOp(255)));
	Cnet::InMemoryDataset train_data(params, pipeline);
	Cnet::read_mnist(train_data, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	Cnet::InMemoryDataset val_data(params, pipeline);
	Cnet::read_mnist(val_data, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	std::shared_ptr<Cnet::Layer> input = Cnet::InputLayer::create(28, 1);
	std::shared_ptr<Cnet::Layer> conv2d_1 = Cnet::Conv2dLayer::create(input, 2, 3,1,Cnet::VALID);
	std::shared_ptr<Cnet::Layer> conv2d_2 = Cnet::Conv2dLayer::create(conv2d_1, 2, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> pool_1 = Cnet::MaxPool2dLayer::create(conv2d_2, 2, 2);
	std::shared_ptr<Cnet::Layer> conv2d_3 = Cnet::Conv2dLayer::create(pool_1, 4, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> conv2d_4 = Cnet::Conv2dLayer::create(conv2d_3, 4, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> pool_2 = Cnet::MaxPool2dLayer::create(conv2d_4, 2, 2);
	std::shared_ptr<Cnet::Layer> fc_layer_1 = Cnet::FcLayer::create(pool_2, 128);
	std::shared_ptr<Cnet::Layer> fc_layer_2 = Cnet::FcLayer::create(fc_layer_1, 10, Cnet::SOFTMAX);
	Cnet::Graph graph(fc_layer_2);
	std::shared_ptr<Cnet::Evaluator> eval = std::make_shared<Cnet::ClassificationEvaluator>();
	Cnet::Estimator estimator(graph, eval,std::unique_ptr<Cnet::Loss>(new Cnet::CrossEntropyLoss), std::unique_ptr<Cnet::Solver>(new Cnet::AdamSolver(0.001)));
	estimator.train(train_data, val_data, 8, 100000, 100, 0, 1000);
	std::cout << "INFO:: (Main) Running test after training..." << std::endl;
	float val = eval->test(graph, val_data);
	std::cout << "INFO:: (Main) Model scored " << val << std::endl;
}

int main(const int argc, const char *const argv[])
{
	Cnet::disable_multithreading();
	//process_program_options(argc, argv);
	train_mnist();
	std::cin.get();
	return 0;
}

