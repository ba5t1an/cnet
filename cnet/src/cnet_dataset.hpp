/**
Implementation of several datasets to train and test neural networks.

@file cnet_dataset.hpp
@author Bastian Schoettle
*/

#ifndef CNET_DATASET_HPP
#define CNET_DATASET_HPP


#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <chrono>
#include <deque>
#include <exception>
#include "cnet_common.hpp"
#include "cnet_preproc.hpp"

namespace Cnet
{

	enum DatasetType
	{
		IN_MEM = 0, ONLINE = 1
	};

	struct DatasetParams
	{
		size_t num_classes;
		bool encode_one_hot;

	};


	class EodException : public std::exception
	{
		virtual const char* what() const throw() override
		{
			return "ERROR:: End of data reached.";
		}
	};

	/*
	Template for Event class
	*/
	template <class T1, class T2> class DataEvent
	{

	private:
		T1 _x;
		T2 _y;

	public:
		DataEvent(T1 x, T2 y)
		{
			_x = x;
			_y = y;
		}

		T1 get_x()
		{
			return _x;
		}

		T2 get_y()
		{
			return _y;
		}
	};

	class Producer
	{

	private:

		std::vector<std::pair<std::string, std::string>> _data;

		DataQueue<DataEvent<Entry, Entry>>& _queue;

		std::shared_ptr<Pipeline> _input_pipeline;

		std::shared_ptr<Pipeline> _label_pipeline;

		DatasetParams _params;
		
		size_t _item_ptr;

		volatile bool _is_running;

		std::unique_ptr<std::thread> _producer_thread;

		inline static void process_image(MatrixRm* matrix, std::shared_ptr<Pipeline>& pipeline)
		{
			if (pipeline)
			{
				pipeline->apply(matrix);
			}
		}

		inline void load_segmentation_data(MatrixRm* input, MatrixRm* label, const std::pair<std::string, std::string>& pair)
		{
			load_image(input, pair.first);
			load_image(label, pair.second);
			process_image(input, _input_pipeline);
			process_image(label, _label_pipeline);
		}

	public:

		Producer(DataQueue<DataEvent<Entry, Entry> >& queue, std::vector<std::pair<std::string, std::string>>& data, DatasetParams& params, std::shared_ptr<Pipeline>& input_pipeline, std::shared_ptr<Pipeline>& label_pipeline) : _queue(queue)
		{
			_params = params;
			_item_ptr = 0;
			_input_pipeline = input_pipeline;
			_label_pipeline = label_pipeline;
			_data = data;
            _is_running = true;
			_producer_thread = std::unique_ptr<std::thread>(new std::thread(&Producer::run, this));
		}

		void run()
		{
            std::cout << "INFO::" <<  "(Dataset) Producer thread (tid=" << std::this_thread::get_id() << ") running." << std::endl;
			/*
			Producer will always provide data if queue size
			is smaller then max size.
			*/
			while (_is_running)
			{
			    if (_item_ptr >= this->_data.size())
				{
					_item_ptr = 0;
				}
				std::pair<std::string, std::string> item = this->_data[_item_ptr++];
				//TODO: Check if segmentation task...this is actually not a proper check...
				if (file_exists(item.second))
				{
					MatrixRm input;
					MatrixRm label;
					load_segmentation_data(&input, &label, item);
					_queue.push(DataEvent<Entry, Entry>(Entry(input, item.first), Entry(label, item.second)));
				}
				else //Apply this for classification/regression tasks
				{
					MatrixRm input;
					load_image(input, item.first);
					process_image(&input, _input_pipeline);
					int label = std::stoi(item.second);
					if (_params.encode_one_hot)
					{
						MatrixRm encoded_label = encode_one_hot(label, _params.num_classes);
						_queue.push(DataEvent<Entry, Entry>(Entry(input, item.first), Entry(encoded_label)));
					}
					else
					{
						MatrixRm scalar = MatrixRm::Zero(1, 1);
						scalar(0, 0) = (float)label;
						_queue.push(DataEvent<Entry, Entry>(Entry(input, item.first), Entry(scalar)));
					}
				}
			}
		}

		void join()
		{
            std::cout << "DEBUG::" <<  "(Dataset) Received join request (tid=" << std::this_thread::get_id() << ")." << std::endl;
			if (_producer_thread)
            {
                std::cout << "DEBUG::" <<  "(Dataset) Joining producer thread (tid=" << std::this_thread::get_id() << ")." << std::endl;
                _is_running = false;
                if(_queue.get_size() == _queue.get_max_size())
                {
                    DataEvent<Entry, Entry> data_event = _queue.pop();
                }
				_producer_thread->join();
			}
            std::cout << "DEBUG::" <<  "(Dataset) Successfully joined producer thread (tid=" << std::this_thread::get_id() << ")." << std::endl;
		}
	};


	template<typename T1, typename T2> class Dataset {

	protected:
		/*
		Internal buffer
		*/
		std::vector<std::pair<T1, T2> > _data_buffer;

		/*
		Counter for current epoch.
		*/
		volatile size_t _current_epoch;

		/*
		If true, dataset will repeat.
		*/
		bool _is_repeating;

		size_t _dataset_size;

		DatasetParams _dataset_params;

		std::shared_ptr<Pipeline> _input_pipeline;
		std::shared_ptr<Pipeline> _label_pipeline;


	public:

		/*
		* Default constructor for class InMemoryDataset
		*/
        explicit Dataset(DatasetParams& params)
		{
			this->_dataset_params = params;
			_current_epoch = 0;
			_is_repeating = false;
		}

		/*
		* Default constructor for class Dataset
		*/
		Dataset(DatasetParams& params, const std::shared_ptr<Pipeline>& input_pipeline, const std::shared_ptr<Pipeline>& label_pipeline)
		{
			this->_dataset_params = params;
			this->_input_pipeline = std::move(input_pipeline);
			this->_label_pipeline = std::move(label_pipeline);
			_current_epoch = 0;
			_is_repeating = false;
		}

		/*
		* Default constructor for class Dataset
		*/
		Dataset(DatasetParams& params, const std::shared_ptr<Pipeline>& input_pipeline)
		{
			this->_dataset_params = params;
			this->_input_pipeline = std::move(input_pipeline);
			_current_epoch = 0;
			_is_repeating = false;
		}

		/*
		* Default constructor for class Dataset
		*/
		Dataset()
		{
			_current_epoch = 0;
			_is_repeating = false;
		}

		/*
		* Default constructor for class Dataset
		*/
		Dataset(const std::shared_ptr<Pipeline>& input_pipeline, const std::shared_ptr<Pipeline>& label_pipeline)
		{
			this->_input_pipeline = std::move(input_pipeline);
			this->_label_pipeline = std::move(label_pipeline);
			_current_epoch = 0;
			_is_repeating = false;
		}

		/*
		* Default constructor for class Dataset
		*/
        explicit Dataset(const std::shared_ptr<Pipeline>& input_pipeline)
		{
			this->_input_pipeline = std::move(input_pipeline);
			_current_epoch = 0;
			_is_repeating = false;
		}


		size_t get_num_classes()
		{
			return _dataset_params.num_classes;
		}

		/*
		Returns the current epoch
		*/
		size_t get_current_epoch()
		{
			return _current_epoch;
		}

		/*
		Returns the dataset size
		*/
		void enabled_auto_repeat()
		{
			_is_repeating = true;
		}

		/*
		Returns next batch of size batch_size
		*/
		virtual void next_batch(std::vector<std::pair<Entry, Entry> >& batch, size_t batch_size) = 0;

		/*
		* Copies the content of the current item in the dataset to the provided instances of data and label as MatrixRm.
		*/
		virtual void next_sample(Entry& data, Entry& label) = 0;


		/*
		Get next Entry (input only) from dataset.
		*/
		virtual void next_sample(Entry& data) = 0;


		/*
		* Splits dataset by ration. Here, ratio is the relative amount t of data that will be used for training.
		* The amount 1 - t will be used for validation.
		*/
		void split(Dataset<T1, T2>& val_data, float ratio)
		{
		    const auto train_amount = (size_t)floor(_data_buffer.size() * ratio);
			val_data._dataset_size = _dataset_size - train_amount;
			_dataset_size = train_amount;

			std::vector<std::pair<T1, T2> > train_buffer(this->_data_buffer.begin(), this->_data_buffer.begin() + train_amount);
			std::vector<std::pair<T1, T2> > val_buffer(this->_data_buffer.begin() + train_amount, this->_data_buffer.end());

			_data_buffer = train_buffer;
			val_data._data_buffer = val_buffer;
		}

		/*
		* Splits dataset by ration. Here, ratio is the relative amount t of data that will be used for training.
		* The amount 1 - t will be used for validation.
		*/
		void shuffle(int seed)
		{
			auto rng = std::default_random_engine(seed);
			std::shuffle(std::begin(_data_buffer), std::end(_data_buffer), rng);
		}



		void shuffle()
		{
			shuffle(1);
		}

		virtual void reset() = 0;

	};


	class InMemoryDataset : public Dataset<Entry, Entry>
	{
	private:
		size_t _item_ptr;

		void init()
		{
			_item_ptr = 0;
		}

	public:

		/*
		* Default constructor for class InMemoryDataset
		*/
        explicit InMemoryDataset(DatasetParams& params) : Dataset(params)
		{
			init();
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		InMemoryDataset(DatasetParams& params, const std::shared_ptr<Pipeline>& input_pipeline) : Dataset(params, input_pipeline)
		{
			init();
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		InMemoryDataset() : Dataset()
		{
			init();
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		InMemoryDataset(std::shared_ptr<Pipeline> input_pipeline, std::shared_ptr<Pipeline> label_pipeline) : Dataset(input_pipeline, label_pipeline)
		{
			init();
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		InMemoryDataset(std::shared_ptr<Pipeline> input_pipeline) : Dataset(input_pipeline)
		{
			init();
		}

		/*
		* Returns next batch of size batch_size
		*/
		void next_batch(std::vector<std::pair<Entry, Entry> >& batch, size_t batch_size) override
		{
			if (_item_ptr >= _data_buffer.size())
			{
				if (!_is_repeating)
				{
					throw EodException();
				}
				_item_ptr = 0;
				++_current_epoch;
			}
			batch.clear();
			bool end_of_epoch = false;
			for (size_t i = 0; i < batch_size; ++i)
			{
				batch.push_back(_data_buffer[_item_ptr++]);
				if (_item_ptr >= _data_buffer.size())
				{
					end_of_epoch = true;
					break;
				}
			}
			if (end_of_epoch)
			{
				++_current_epoch;
			}
		}


		/*
		* Copies the content of the current item in the dataset to the provided instances of data and label as MatrixRm.
		*/
		void next_sample(Entry& data, Entry& label) override
		{
			if (_item_ptr == _data_buffer.size())
			{
				if (!_is_repeating)
				{
					throw EodException();
				}
				_item_ptr = 0;
				++_current_epoch;
			}

			std::pair<Entry, Entry> item = _data_buffer[_item_ptr++];
			data = item.first;
			label = item.second;
		}


		/*
		* Get next Entry (input only) from dataset.
		*/
		void next_sample(Entry& data) override
		{
			if (_item_ptr == _data_buffer.size())
			{
				if (!_is_repeating)
				{
					throw EodException();
				}
				_item_ptr = 0;
				++_current_epoch;
			}
			std::pair<Entry, Entry> item = _data_buffer[_item_ptr++];
			data = item.first;
		}


		/*
		* Method to add Entry
		*/
		void add_sample(Entry input, const size_t label)
		{
			if (_input_pipeline)
			{
				_input_pipeline->apply(input.data());
			}
			MatrixRm label_matrix;
			if (_dataset_params.encode_one_hot)
			{
				encode_one_hot(label_matrix, label, _dataset_params.num_classes);
			}
			else
			{
				label_matrix = MatrixRm(1, 1);
				label_matrix(0, 0) = label;
			}
			_data_buffer.emplace_back(input, Entry(label_matrix));
		}


		/*
		* Alternative method to add Entry
		*/
		void add_sample(Entry input, Entry label)
		{
            std::cout << "DEBUG::" << "(Dataset) Loading image " << input.file_path() << std::endl;
			if (_input_pipeline)
			{
				_input_pipeline->apply(input.data());
			}
			if (_label_pipeline)
			{
				_label_pipeline->apply(label.data());
			}
			_data_buffer.emplace_back(input, label);
		}

		void reset() override
		{
			_item_ptr = 0;
		}

	};


	/*
	OnlineDataset class for training and test data. Loads data streight into system's memory.
	NOTE: If dataset is large and does not fit into memory all at once...consider using a different one.
	*/
	class OnlineDataset : public Dataset<std::string, std::string>
	{
	private:
		size_t _queue_size;
		DataQueue<DataEvent<Entry, Entry> > _queue;
		bool _is_producer_initialized;
		size_t _items_left;
		size_t _num_threads;
		std::vector<std::unique_ptr<Producer> > _producers;
		size_t _item_ptr;

		void init(size_t num_threads)
		{
			_queue_size = INITIAL_QUEUE_SIZE;
			_is_producer_initialized = false;
			_items_left = 0;
			_item_ptr = 0;
			_num_threads = num_threads;
            _producers.resize(num_threads);
		}

		void initialize_producers()
		{
			size_t cnt = 0;
			std::vector< std::vector<std::pair<std::string, std::string> > > splits(_num_threads);
			for (size_t i = 0; i < _data_buffer.size(); ++i)
			{
				splits[cnt].push_back(_data_buffer[i]);
				if (++cnt == _num_threads)
				{
					cnt = 0;
				}
			}
			for (size_t i = 0; i < _num_threads; ++i)
			{
                _producers[i] = std::unique_ptr<Producer>(new Producer(_queue, splits[i], _dataset_params, _input_pipeline, _label_pipeline));
			}
            std::cout << "DEBUG::" << "(Dataset) Fetching " << _queue_size << " items..." << std::endl;
			while (_queue.get_size() < _queue_size)
			{
				std::this_thread::sleep_for(std::chrono::nanoseconds(THREAD_RETRY_DELAY));
			}
            std::cout << "DEBUG::" << "(Dataset) Fetched " << _queue.get_size() << " items." << std::endl;
		}

		void pop_next_pair(Entry& input, Entry& label)
		{
			DataEvent<Entry, Entry> data_event = _queue.pop();
			input = data_event.get_x();
			label = data_event.get_y();
		}

		void pop_next_x(Entry& input)
		{
			DataEvent<Entry, Entry> data_event = _queue.pop();
			input = data_event.get_x();
		}

		void pop_next_batch(std::vector<std::pair<Entry, Entry> >& batch, size_t batch_size)
		{
			for (size_t i = 0; i < batch_size; ++i)
			{
				Entry input;
				Entry label;
				pop_next_pair(input, label);
				batch.emplace_back(input, label);
			}
		}


	public:

		/*
		* Default constructor for class
		*/
		OnlineDataset(DatasetParams& params, size_t num_threads = 1) : Dataset(params)
		{
			init(num_threads);
		}

		/*
		* Default constructor for class
		*/
		OnlineDataset(DatasetParams& params, std::shared_ptr<Pipeline> input_pipeline, size_t num_threads = 1) : Dataset(params, input_pipeline)
		{
			init(num_threads);
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		OnlineDataset(size_t num_threads = 1) : Dataset()
		{
			init(num_threads);
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		OnlineDataset(std::shared_ptr<Pipeline> input_pipeline, std::shared_ptr<Pipeline> label_pipeline, size_t num_threads = 1) : Dataset(input_pipeline, label_pipeline)
		{
			init(num_threads);
		}

		/*
		* Default constructor for class InMemoryDataset
		*/
		OnlineDataset(std::shared_ptr<Pipeline> input_pipeline, size_t num_threads = 1) : Dataset(input_pipeline)
		{
			init(num_threads);
		}

		void add_sample(std::string input, std::string label)
		{
			_data_buffer.emplace_back(input, label);
		}


		void next_sample(Entry& input, Entry& label) override
		{
			if (!_is_producer_initialized)
			{
				/*
				If not initialized, initialize producer threads
				*/
				initialize_producers();
				_is_producer_initialized = true;
			}
			if (_is_repeating)
			{
				pop_next_pair(input, label);
			}
			else
			{
				if (_item_ptr++ == _data_buffer.size())
				{
					++_current_epoch;
					throw EodException();
				}
				pop_next_pair(input, label);
			}
		}


		void next_sample(Entry& input) override
		{
			if (!_is_producer_initialized)
			{
				/*
				If not initialized, initialize producer threads
				*/
				initialize_producers();
				_is_producer_initialized = true;
			}
			if (_item_ptr++ == _data_buffer.size())
			{
				if (_is_repeating)
				{
					++_current_epoch;
					_item_ptr = 0;
					pop_next_x(input);
				}
				else
				{
					++_current_epoch;
					throw EodException();
				}
			}
			else
			{
				pop_next_x(input);
			}
		}


		void next_batch(std::vector<std::pair<Entry, Entry>>& batch, size_t batch_size) override
		{
			if (!_is_producer_initialized)
			{
				/*
				If not initialized, initialize producer threads
				*/
				initialize_producers();
				_is_producer_initialized = true;
			}
			if (_item_ptr++ == _data_buffer.size())
			{
				if (_is_repeating)
				{
					++_current_epoch;
					_item_ptr = 0;
				}
				else
				{
					++_current_epoch;
					throw EodException();
				}	
			}
			batch.clear();
			size_t actual_batch_size = batch_size;
			if ((_item_ptr + batch_size) >= _data_buffer.size())
			{
				actual_batch_size = _data_buffer.size() - _item_ptr;
				_item_ptr = _data_buffer.size();
			}
			pop_next_batch(batch, actual_batch_size);
		}


		size_t get_queued_items()
		{
			return _queue.get_size();
		}

		void reset() override
		{
			_item_ptr = 0;
		}

		~OnlineDataset()
		{
			for (size_t i = 0; i < _num_threads; ++i)
			{
                std::cout << "DEBUG::" <<  "(Dataset) Trying to join threads, num_threads = " << std::to_string(_num_threads) <<", thread_vec_items = " << std::to_string(_producers.size()) << "..." << std::endl;
                if (_producers[i])
				{
                    std::cout << "DEBUG::" <<  "(Dataset) Trying to join thread = " << std::to_string(i) << "..." << std::endl;
					_producers[i]->join();
				}
			}
		}
	};
}

#endif
