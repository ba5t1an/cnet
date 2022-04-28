#ifndef CNET_EVALUATOR_HPP
#define CNET_EVALUATOR_HPP

#include "cnet_common.hpp"
#include "cnet_graph.hpp"
#include "cnet_dataset.hpp"
#include "cnet_imglogger.hpp"

namespace Cnet
{


	class Evaluator
	{
	protected:
		std::unique_ptr<ImageLogger> _image_logger;

	public:
		Evaluator()
		{
		}

		//~Evaluator() = default;

		void attach_logger(std::string location)
		{
			_image_logger = std::unique_ptr<ImageLogger>(new ImageLogger(location));
		}

		virtual float test(Graph& g, Dataset<Entry, Entry>& dataset) = 0;
		virtual float test(Graph& g, Dataset<std::string, std::string>& dataset) = 0;

	};

	class ClassificationEvaluator : public Evaluator
	{
	private:

		float _threshold;

		/*
		* Determine if a single output is abvoe or below BIN_THRESHOLD.
		* Used in binary cassification with only one output node.
		*/
		template<class T1, class T2> inline float binary_accuracy(Graph& g, Dataset<T1, T2>& dataset)
		{
			Entry label;
			Entry data;
			MatrixRm output;
			size_t score = 0;
			size_t total = 0;
			dataset.reset();
			do
			{
				try
				{
					dataset.next_sample(data, label);
				}
				catch (EodException& e)
				{
					(void)e;
					break;
				}

				g.forward(data.data(), &output);
				if (output(0, 0) >= _threshold && (*label.data())(0, 0) == 1)
				{
					score += 1;
				} 
				else if (output(0, 0) < _threshold && (*label.data())(0, 0) == 0)
				{
					score += 1;
				}
				else
				{
					if (_image_logger)
					{
						std::string image_name = data.file_name() + "_" + std::to_string(output(0, 0)) + "_" + "_" + std::to_string((*label.data())(0, 0)) + ".png";
						_image_logger->log_image(data.data(), "err", image_name);
					}
				}
				++total;
			} while (true);
			return score / (float)total;
		}

		/*
		* Determine if a the argmax of an output matches the target label.
		* Used in multiclass classification. Precisly if more than one output node is present.
		*/
		template<class T1, class T2> inline float multiclass_accuracy(Graph& g, Dataset<T1, T2>& dataset)
		{
			Entry label;
			Entry data;
			MatrixRm output;
			size_t score = 0;
			size_t total = 0;
			dataset.reset();
			do
			{
				try
				{
					dataset.next_sample(data, label);
				}
				catch (EodException& e)
				{
					(void)e;
					break;
				}

				g.forward(data.data(), &output);
				size_t max_idx = argmax(output);
				size_t expected_idx = argmax(label.data());
				if (max_idx == expected_idx)
				{
					score += 1;
				}
				else
				{
					if (_image_logger)
					{
						std::string image_name = data.file_name() + "_" + std::to_string(output(0, max_idx)) + "_" + std::to_string(max_idx) + "_" + std::to_string(expected_idx) + ".png";
						_image_logger->log_image(data.data(), "err", image_name);
					}
				}
				++total;
			} while (true);
			return score / (float)total;
		}

		template<class T1, class T2> float test_func(Graph& g, Dataset<T1, T2>& dataset) 
		{
			float score = 0.f;
			if (g.get_output_cols() == 1 && g.get_output_rows() == 1)
			{
				score = binary_accuracy(g, dataset);
			}
			else
			{
				score = multiclass_accuracy(g, dataset);
			}
			return score;
		}

	public:

		ClassificationEvaluator() : Evaluator()
		{
			_threshold = 0.5f;
		}

		ClassificationEvaluator(float threshold)
		{
			_threshold = threshold;
		}

		//~ClassificationEvaluator() = default;


		float test(Graph& g, Dataset<Entry, Entry>& dataset) override
		{
			float score = test_func(g, dataset);
			return score;
		}

		float test(Graph& g, Dataset<std::string, std::string>& dataset) override
		{
			float score = test_func(g, dataset);
			return score;
		}

	};



	
}

#endif