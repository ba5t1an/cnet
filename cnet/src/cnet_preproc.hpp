/**
Implementation of serveral proprocessing functions defined as operations.
A preprocessing pipeline is defined by a flow graph of such operations.

@file cnet_preproc.hpp
@author Bastian Schoettle EN RC PREC
*/

#ifndef CNET_PREPROC_HPP
#define CNET_PREPROC_HPP

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include "cnet_common.hpp"

namespace Cnet 
{

	/*
	Operation base class acts as an interface for any operation.
	NOTE: A matrix passed to apply(...) will be modified in place.
	*/
	class Operation
	{

	public:

		/*
		Applies defined operation to the input matrix.
		*/
		virtual void apply(MatrixRm* data) = 0;
	};

	/*
	Performs channel-wise normalization using moments of every channel.
	*/
	class PerChannelStandardizationOp: public Operation{


		PerChannelStandardizationOp() : Operation()
		{}
		
		void apply(MatrixRm* data) override
		{
			for (unsigned int i = 0; i < data->rows(); i++)
			{
				float mean = 0;
				unsigned int m = data->cols();
				mean += data->row(i).sum() / (float)m;
				float std_dev = 0;
				for (unsigned int j = 0; j < m; ++j) {
					std_dev += pow((*data)(i, j) - mean, 2.f);
				}
				std_dev /= m - 1;
				std_dev = sqrt(std_dev);
				for (unsigned int j = 0; j < m; ++j) {
					(*data)(i, j) = ((*data)(i, j) - (float)mean) / std::max(std_dev, 1.f / m);
				}
			}
		}
	};


	/*
	Performs normalization using moments of first channel.
	*/
	class PerImagStandardizationOp : public Operation {

	public:

		PerImagStandardizationOp() : Operation()
		{}
		void apply(MatrixRm* data) override
		{
				float mean = 0;
				unsigned int m = data->cols();
				mean += data->sum() / (float)m;
				float std_dev = 0;
				for (unsigned int i = 0; i < data->rows(); ++i)
				{
					for (unsigned int j = 0; j < m; ++j) {
						std_dev += pow((*data)(i, j) - mean, 2.f);
					}
				}
				std_dev /= m - 1;
				std_dev = sqrt(std_dev);
				for (unsigned int i = 0; i < data->rows(); i++)
				{
					for (unsigned int j = 0; j < m; ++j) {
						(*data)(i, j) = ((*data)(i, j) - (float)mean) / std::max(std_dev, 1.f / m);
					}
				}
		}
	};

	/*
	Scale operation. 
	*/
	class ScaleOp : public Operation {
	private:
		float _scale_factor;

	public:

		ScaleOp(float scale_factor) : Operation()
		{
			_scale_factor = scale_factor;
		}

		void apply(MatrixRm* data) override
		{
			*data /= _scale_factor;
		}
	};

	/*
	Scale operation.
	*/
	class RoundOp : public Operation {
	private:

	public:

		RoundOp() : Operation()
		{
		}

		void apply(MatrixRm* data) override
		{
			for (unsigned int i = 0; i < data->cols(); i++)
			{
				(*data)(0, i) = round((*data)(0, i));
			}
		}
	};

	/*
	Resize operation.
	*/
	class ResizeOp : public Operation {

	private:
		int _trg_size;

		bool is_int(float d)
		{
			float dummy;
			return modf(d, &dummy) == 0.0;
		}

	public:

		ResizeOp(int trg_size) : Operation()
		{
			_trg_size = trg_size;
		}

		void apply(MatrixRm* data) override
		{
			size_t input_width = sqrt(data->cols());
			if (!is_int(sqrt(data->cols())))
			{
				input_width = sqrt(data->cols() + 1);
			}
			MatrixRm tmp = MatrixRmMap(data->data(), input_width, input_width);
			cv::Mat cv_resized;
			cv::resize(Cnet::eigen2cv(tmp), cv_resized, cv::Size(_trg_size, _trg_size), 0, 0, cv::INTER_CUBIC);
			Cnet::eigen2cv(tmp) = cv_resized;
			*data = MatrixRmMap(tmp.data(), 1, _trg_size*_trg_size);
		}
	};


	class CropOp : public Operation 
	{
	public:
		CropOp(size_t trg_width) : Operation()
		{
			_trg_width = trg_width;
		}

		void apply(MatrixRm* data) override
		{
			auto org_width = (size_t) sqrt(data->cols());
			MatrixRm cropped;
			size_t offset = org_width - _trg_width;
			crop(data, &cropped, offset);
			*data = cropped;
		}
	private:
		size_t _trg_width;
	};

	class PadOp : public Operation
	{
	public:
		PadOp(size_t padding) : Operation()
		{
			_padding = padding;
		}

		void apply(MatrixRm* data) override
		{
			MatrixRm padded;
			pad(data, &padded, _padding);
			*data = padded;
		}
	private:
		size_t _padding;
	};

	/*
	Pipeline class
	*/
	class Pipeline
	{
	public:
		
		/*
		Pipeline constructor
		*/
		Pipeline()
		{
		}
		//~Pipeline() = default;

		/*
		Add operation to the pipeline
		*/
		void add_operation(std::unique_ptr<Operation> op)
		{
			this->_pipeline.push_back(std::move(op));
		}

		/*
		Apply operation pipeline to input matrix
		*/
		void apply(MatrixRm* matrix)
		{
			for (size_t i = 0; i < _pipeline.size(); ++i)
			{
				_pipeline[i]->apply(matrix);
			}
		}

	private:
		std::vector<std::unique_ptr<Operation> > _pipeline;
	};

}

#endif

