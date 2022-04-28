/**
Implementation of an estimator to train neural networks.

@file cnet_estimator.hpp
@author Bastian Schoettle
*/

#ifndef CNET_ESTIMATOR_HPP
#define CNET_ESTIMATOR_HPP

#include <memory>
#include <limits>
#include <iomanip>
#include <chrono>
#include "cnet_common.hpp"
#include "cnet_graph.hpp"
#include "cnet_loss.hpp"
#include "cnet_solver2.hpp"
#include "cnet_dataset.hpp"
#include "cnet_saver.hpp"
#include "cnet_vlogger.hpp"
#include "cnet_evaluator.hpp"
#include "cnet_imglogger.hpp"

namespace Cnet
{

	/*
	Forward declaration for class GraphThread
	*/
	class GraphThread;

	/*
	Interface for thread control
	*/
	class GraphThreadController
	{
	protected:
		volatile bool _is_training;
		volatile size_t _current_batch_items;
		std::vector<std::unique_ptr<GraphThread> > _graph_threads;
		std::mutex _mutex;

	public:

		GraphThreadController()
		{
			_is_training = true;
			_current_batch_items = 0;
		}

		void on_item_complete()
		{
			std::unique_lock<std::mutex> lock(_mutex);
			++_current_batch_items;
			//std::cout << "DEBUG:: (Estimator) Thread completed, item_cnt = " << std::to_string(_current_batch_items) << std::endl;
		}

		bool is_training()
		{
			return _is_training;
		}

		void set_training(bool is_training)
		{
			_is_training = is_training;
		}


	};

	/*
	* Graph thread class
	*/
	class GraphThread
	{
	private:

		Graph _g;
		DataQueue<std::pair<Entry, Entry> *> _queue;
		GraphThreadController *_controller;
		std::unique_ptr<Loss> _loss;
		std::unique_ptr<std::thread> _thread;
		Graph &_master;

	public:

		GraphThread(GraphThreadController *controller, Graph &master, Loss *loss) : _master(master)
		{
			_master.clone(&_g);
			_controller = controller;
			_loss = loss->clone();
			_queue.set_max_size(INITIAL_QUEUE_SIZE);
			_thread = std::unique_ptr<std::thread>(new std::thread(&GraphThread::run, this));
		}

		void join()
		{
			_thread->join();
		}

		void provide(std::pair<Entry, Entry> *data)
		{
			_queue.push(data);
			//std::cout << "DEBUG:: (Estimator) Thread (TID = " << std::this_thread::get_id() <<") got " << _queue.get_size()  << " items."<< std::endl;
		}

		void run()
		{
			std::cout << "DEBUG::" << "(Estimator) Started graph thread (TID = " << std::this_thread::get_id() << ")" << std::endl;
			while (_controller->is_training()) {
				std::pair<Entry, Entry> *data;
				bool success = _queue.try_pop(data, std::chrono::milliseconds(1));
				MatrixRm output;
				if (success) {
					//auto start = std::chrono::high_resolution_clock::now();

					this->_g.forward((*data).first.data(), &output, true);
					MatrixRm deltas = _loss->derivative(&output, (*data).second.data());
					_g.backward(&deltas);
					_controller->on_item_complete();
					//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
				}
			}
		}

		void gather_gradients(Graph &master)
		{
			master.gather_and_reset_gradients(&_g);
		}

		void sync_weights(Graph &master)
		{
			_g.sync_weights(&master);
		}

	};


	/*
	* Class Estimator which provides the functionality to train and test a Graph instance.
	* NOTE: If this class is used to train a Graph, it will modify it's weights using an instance of Solver
	*/
	class Estimator : public GraphThreadController
	{
	private:
		/*
		* Graph instance to train and/or test.
		*/
		Graph &_graph;

		/*
		*Evaluator for test function
		*/
		std::shared_ptr<Cnet::Evaluator> _evaluator;

		/*
		* Loss function to optimize by a Solver instance.
		*/
		std::unique_ptr<Loss> _loss;
		/*
		* Solver instance to update weights of the graph.
		*/
		std::unique_ptr<Solver> _solver;
		/*
		* Saver instance to save model.
		*/
		std::unique_ptr<Saver> _saver;

		/*
		Visitor to retrieve layer information
		*/
		std::unique_ptr<Visitor> _visitor;

		/*
		Iteration for the visitor to be applied
		*/
		size_t _visit_at_iter;

		/*
		Number of visits at iteration _visit_at_iter
		*/
		size_t _num_visits;

		/*
		Best achived score
		*/
		float _best_score;
		/*
		* Forward time variable
		*/
		double _fwd_time;

		/*
		* Number of threads
		*/
		size_t _num_threads;

		std::unique_ptr<ImageLogger> _img_logger;

		size_t _img_log_iter;

		size_t _num_log_items;

		bool _log_input;

		bool _log_output;

		bool _use_argmax;

		bool _log_label;


		void log_images(std::vector<std::pair<Entry, Entry> > &train_batch, size_t num_items, size_t current_iter, bool use_argmax = false)
		{
			if (_log_input || _log_output || _log_label)
			{
				for (unsigned int i = 0; i < num_items; ++i) {
					if (_log_output)
					{
						MatrixRm outputs;
						this->_graph.forward(train_batch[i].first.data(), &outputs, false);
						if (use_argmax)
						{
							MatrixRm argmaxed;
							depthwise_argmax(outputs, argmaxed);
							_img_logger->log_image(&argmaxed, "output", train_batch[i].first.file_name(), current_iter);
						}
						else
						{
							_img_logger->log_image(&outputs, "output", train_batch[i].first.file_name(), current_iter);
						}
					}
					if (_log_input)
					{
						_img_logger->log_image(train_batch[i].first.data(), "input", train_batch[i].first.file_name(), current_iter);
					}
					if (_log_label)
					{
						_img_logger->log_image(train_batch[i].second.data(), "label", train_batch[i].second.file_name(), current_iter);
					}
				}
			}
		}

		/*
		* This will forward batch_size samples and calculate the gradients sequentially for each sample.
		* Here, gradients will be accumulated. When batch_size samples have been for- and backwarded, a weight update
		* will be performed by the specified solver instance.
		*/
		void compute_batch_update(std::vector<std::pair<Entry, Entry> > &train_batch)
		{
			MatrixRm outputs;
			auto start = std::chrono::high_resolution_clock::now();
			for (unsigned int i = 0; i < train_batch.size(); ++i) {
				this->_graph.forward(train_batch[i].first.data(), &outputs, true);
				MatrixRm deltas = this->_loss->derivative(&outputs, train_batch[i].second.data());
				this->_graph.backward(&deltas);
			}
			float adjusted_lr = (1.f / train_batch.size()) * this->_solver->get_lr();
			this->_solver->optimize(this->_graph, adjusted_lr);
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - start).count();
			//std::cout << "DEBUG::" << "(Estimator) processing batch took " << std::to_string(duration) << "ms" << std::endl;
		}

		/*
		* This will forward batch_size samples and calculate the gradients sequentially for each sample.
		* Here, gradients will be accumulated. When batch_size samples have been for- and backwarded, a weight update
		* will be performed by the specified solver instance.
		*/
		void compute_batch_update_parallel(std::vector<std::pair<Entry, Entry> > &train_batch)
		{
			this->_current_batch_items = 0;
			MatrixRm outputs;
			auto start = std::chrono::high_resolution_clock::now();
			size_t thread_cnt = 0;
			for (size_t i = 0; i < train_batch.size(); i++) {
				_graph_threads[thread_cnt++]->provide(&train_batch[i]);
				if (thread_cnt == _num_threads) {
					thread_cnt = 0;
				}
			}
			do {
				std::this_thread::sleep_for(std::chrono::nanoseconds(THREAD_RETRY_DELAY));
			} while (_current_batch_items < train_batch.size());
			for (size_t i = 0; i < _num_threads; ++i) {
				_graph_threads[i]->gather_gradients(_graph);
			}
			float adjusted_lr = (1.f / train_batch.size()) * this->_solver->get_lr();
			this->_solver->optimize(_graph, adjusted_lr);
			for (size_t i = 0; i < _num_threads; ++i) {
				_graph_threads[i]->sync_weights(_graph);
			}
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - start).count();
			std::cout << "DEBUG::" << "(Estimator) Processing batch took " << std::to_string(duration) << "ms." << std::endl;
		}

		/*
		* Forwards batch_size samples and accumulates the error.
		*/
		float compute_batch_loss(std::vector<std::pair<Entry, Entry> > &test_batch)
		{
			float error = 0.;
			MatrixRm outputs;
			this->_fwd_time = 0.f;
			for (unsigned int i = 0; i < test_batch.size(); ++i) {
				auto start = std::chrono::high_resolution_clock::now();
				this->_graph.forward(test_batch[i].first.data(), &outputs);
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::high_resolution_clock::now() - start).count();
				this->_fwd_time += duration;
				error += this->_loss->calculate(&outputs, test_batch[i].second.data());
			}
			this->_fwd_time /= (float)test_batch.size();
			return error / (float)test_batch.size();
		}

		template<class T1, class T2>
		void save_model(Dataset<T1, T2> &train_data, size_t batch_size, size_t current_iter)
		{
			if (_saver) {
				size_t current_epoch = train_data.get_current_epoch();
				this->_saver->store(_graph);
				std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size << "]: "
					<< current_iter << ", Saved model to " << _saver->get_model_path();
			}
		}

		void init_parallel()
		{
			//_queue.set_max_size(batch_size);
			_is_training = true;
			for (size_t i = 0; i < _num_threads; ++i) {
				_graph_threads.push_back(std::unique_ptr<GraphThread>(new GraphThread(this, _graph, _loss.get())));
			}
		}


	public:

		/*
		* Constructor for estimator.
		*/
		Estimator(Graph &graph, std::shared_ptr<Cnet::Evaluator> evaluator, std::unique_ptr<Loss> loss,
			std::unique_ptr<Solver> solver, size_t num_threads = 0) : GraphThreadController(), _graph(graph)
		{
			this->_loss = std::move(loss);
			this->_solver = std::move(solver);
			this->_saver = nullptr;
			this->_visitor = nullptr;
			this->_visit_at_iter = 0;
			this->_best_score = 0.f;
			this->_evaluator = evaluator;
			this->_num_threads = num_threads;
		}

		/*
		* Constructor for estimator.
		*/
		Estimator(Graph &graph, std::shared_ptr<Cnet::Evaluator> evaluator, std::unique_ptr<Loss> loss,
			std::unique_ptr<Solver> solver, std::unique_ptr<Saver> saver, size_t num_threads = 0)
			: GraphThreadController(), _graph(graph)
		{
			this->_loss = std::move(loss);
			this->_solver = std::move(solver);
			this->_saver = std::move(saver);
			this->_visitor = nullptr;
			this->_visit_at_iter = 0;
			this->_best_score = 0.f;
			this->_evaluator = evaluator;
			this->_num_threads = num_threads;
		}

		void attach_image_logger(std::string location, bool log_input, bool log_output, bool log_label, size_t log_iter,
			size_t num_log_items, bool use_argmax = false)
		{
			this->_img_logger = std::unique_ptr<ImageLogger>(new ImageLogger(location));
			_log_input = log_input;
			_log_output = log_output;
			_log_label = log_label;
			_img_log_iter = log_iter;
			_num_log_items = num_log_items;
			_use_argmax = use_argmax;
		}

		void attach_visitor(std::unique_ptr<Visitor> visitor, size_t at_iteration, size_t num_visits = 1)
		{
			_visit_at_iter = at_iteration;
			_visitor = std::move(visitor);
			_num_visits = num_visits;
		}


		/*
		Template method to train graph.
		*/
		template<class T1, class T2>
		void train(Dataset<T1, T2> &train_data, Dataset<T1, T2> &val_data, const size_t batch_size,
			const size_t max_iter = 10000, const size_t display_iter = 1, const size_t save_iter = 10000,
			const size_t test_iter = 1000)
		{
			if (_num_threads > 0) {
				std::cout << "INFO::" << "(Estimator) Initializing parallel training -> num_threads: "
					<< std::to_string(_num_threads) << std::endl;
				init_parallel();
			}

			std::cout << "INFO::" << "(Estimator) Started training -> batch_size: " << batch_size << ", max_iter: "
				<< max_iter << ", test_iter: " << test_iter << ", display_iter: " << display_iter
				<< ", save_iter: " << save_iter << std::endl;
			if (!this->_solver) {
				std::cout << "ERROR::" << "(Estimator) Initializing parallel training -> num_threads: "
					<< std::to_string(_num_threads) << std::endl;
				throw std::runtime_error("(Estimator) Unable to train model. No solver defined...");
			}
			size_t current_iter = 0;
			size_t current_epoch = 0;
			bool skip_batch = false;
			std::vector<std::pair<Entry, Entry>> current_batch;
			do {
				try {
					if (!skip_batch) {
						train_data.next_batch(current_batch, batch_size);
					}
					else {
						skip_batch = false;
					}
				}

				catch (EodException &e) {
					(void)e;
					std::cout << "ERROR:: (Estimator) End of dataset reached." << std::endl;
					break;
				}

				if (_num_threads > 0) {
					compute_batch_update_parallel(current_batch);
				}
				else {
					compute_batch_update(current_batch);
				}
				current_epoch = train_data.get_current_epoch();
				++current_iter;
				if (display_iter > 0 && current_iter % display_iter == 0) {
					double error = compute_batch_loss(current_batch);
					size_t current_epoch = train_data.get_current_epoch();
					std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size
						<< "]: " << current_iter << ", Fwd: " << this->_fwd_time << "ms, Loss: "
						<< error << std::endl;
				}
				if (test_iter > 0 && current_iter % test_iter == 0) {
					if (_evaluator)
					{
						std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size
							<< "]: " << current_iter << ", Testing model...";
						float score = _evaluator->test(_graph, val_data);
						std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size
							<< "]: " << current_iter << ", Model scored: " << std::to_string(score) << std::endl;;
						if (score > _best_score) {
							std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size
								<< "]: " << current_iter << ", Model scored: " << std::to_string(score)
								<< ", Recent best score was " << std::to_string(_best_score) << std::endl;;
							_best_score = score;
							save_model(train_data, batch_size, current_iter);
						}
					}
				}
				if (save_iter > 0 && current_iter % save_iter == 0) {
					save_model(train_data, batch_size, current_iter);
				}
				if (_visit_at_iter > 0 && current_iter % _visit_at_iter == 0) {
					if (_visitor) {
						std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size
							<< "]: " << current_iter << ", Applying visitor to graph..." << std::endl;;
						size_t actual_visits = batch_size;
						if (_num_visits < batch_size) {
							actual_visits = _num_visits;
						}
						for (size_t i = 0; i < actual_visits; ++i) {
							_visitor->set_current_iter(current_iter);
							_graph.visit(current_batch[i].first.data(), _visitor.get());
						}
						skip_batch = true;
					}

				}
				if (_img_logger && current_iter % _img_log_iter == 0) {
					std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size
						<< "]: " << current_iter << ", Logging images..." << std::endl;;
					size_t actual_items = _num_log_items;
					if (_num_log_items > batch_size) {
						actual_items = batch_size;
					}
					log_images(current_batch, actual_items, current_iter, _use_argmax);
					skip_batch = true;
				}

			} while (current_iter < max_iter);
			if (_num_threads > 0) {
				_is_training = false;
				for (size_t i = 0; i < _num_threads; i++) {
					_graph_threads[i]->join();
				}
			}
			std::cout << "INFO::" << "(Estimator) Epoch: " << current_epoch << ", Step[" << batch_size << "]: "
				<< current_iter << ", Max iteration reached. Training done." << std::endl;;
		}

	};


}

#endif