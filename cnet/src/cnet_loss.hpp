/**
Implementation of common loss function.

@file cnet_loss.hpp
@author Bastian Schoettle EN RC PREC
*/

#ifndef CNET_LOSS_HPP
#define CNET_LOSS_HPP

#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include "cnet_common.hpp"

#define fraction float(1e-10)

namespace Cnet
{

	class Loss
	{

	public:
		virtual float calculate(MatrixRm* data, MatrixRm* targets) = 0;
		virtual MatrixRm derivative(MatrixRm* data, MatrixRm* targets) = 0;

		virtual std::unique_ptr<Loss> clone() = 0;
	};

	class MseLoss : public Loss
	{
	public:

		float calculate(MatrixRm* data, MatrixRm* targets) override 
		{
			assert(data->cols() == targets->cols() && data->rows() == targets->rows());
            return ((*data - *targets).squaredNorm()) / (targets->cols()*2);
		}

		MatrixRm derivative(MatrixRm* data, MatrixRm* targets) override 
		{
			assert(data->cols() == targets->cols() && data->rows() == targets->rows());
            return (*data - *targets);
		}

        std::unique_ptr<Loss> clone() override
        {
            return std::unique_ptr<MseLoss>(new MseLoss());
        }

	};

	class BinaryCrossEntropyLoss : public Loss
	{
	public:

		float calculate(MatrixRm* data, MatrixRm* targets) override 
		{
			assert(data->cols() == 1 && data->rows() == 1 && targets->rows() == 1 && targets->cols() == 1);
            float stable_target = (*targets)(0, 0) - fraction;
			float stable_output = (*data)(0, 0) - fraction;
			return -(stable_target*log(stable_output) + (1 - stable_target)*log(1 - stable_output));
		}

		MatrixRm derivative(MatrixRm* data, MatrixRm* targets) override 
		{
			assert(data->cols() == 1 && data->rows() == 1 && targets->rows() == 1 && targets->cols() == 1);
            MatrixRm deltas(1, 1);
			deltas(0, 0) = ((*data)(0, 0) - (*targets)(0, 0));
			return deltas;
		}

        std::unique_ptr<Loss> clone() override
        {
            return std::unique_ptr<BinaryCrossEntropyLoss>(new BinaryCrossEntropyLoss());
        }

	};


	class CrossEntropyLoss : public Loss
	{
	private:

		static inline float multiclass_cross_entropy(MatrixRm* outputs, MatrixRm* targets)
		{
			float sum = 0.f;
			for (int i = 0; i < outputs->cols(); i++)
			{
				sum += (*targets)(0, i) * log((*outputs)(0, i));
			}
			return -1 * sum;
		}

	public:

		float calculate(MatrixRm* data, MatrixRm* targets) override 
		{
            return multiclass_cross_entropy(data, targets);
		}


		MatrixRm derivative(MatrixRm* data, MatrixRm* targets) override
		{
			assert(data->cols() == targets->cols() && data->rows() == targets->rows());
            return (*data - *targets);
		}

        std::unique_ptr<Loss> clone() override
        {
            return std::unique_ptr<CrossEntropyLoss>(new CrossEntropyLoss());
        }
	};

	class SparseSoftmaxCrossEntropyLoss : public Loss
	{

	public:


		float calculate(MatrixRm* data, MatrixRm* targets) override 
		{
			assert(data->cols() == targets->cols() && targets->rows() == 1);
            MatrixRm softmaxed = MatrixRm::Zero(data->rows(), data->cols());
			for (int i = 0; i < data->cols(); i++)
			{
				MatrixRm chunk = data->block(0, i, data->rows(), 1);
				MatrixRm normalized_data = chunk.array() - chunk.maxCoeff();
				float exp_sum = 0.f;
				for (int row = 0; row < chunk.rows(); ++row)
				{
					exp_sum += exp(normalized_data(row, 0));
				}

				for (int row = 0; row < chunk.rows(); ++row)
				{
					softmaxed(row, i) = exp(normalized_data(row, 0)) / exp_sum;
				}
			}
			float sum = 0.f;
			for (unsigned int i = 0; i < targets->cols(); i++)
			{
				sum += log(std::max(softmaxed(round((*targets)(0, i)), i), (float) 1e-15));
			}
			return - ((1/ (float)targets->cols()) * sum);
		}


		MatrixRm derivative(MatrixRm* data, MatrixRm* targets) override 
		{
			assert(data->cols() == targets->cols() && targets->rows() == 1);
            MatrixRm softmaxed = MatrixRm::Zero(data->rows(), data->cols());
			for (int i = 0; i < data->cols(); i++)
			{
				MatrixRm chunk = data->block(0, i, data->rows(), 1);
				MatrixRm normalized_data = chunk.array() - chunk.maxCoeff();
				float exp_sum = 0.f;
				for (int row = 0; row < chunk.rows(); ++row)
				{
					exp_sum += exp(normalized_data(row, 0));
				}

				for (int row = 0; row < chunk.rows(); ++row)
				{
					softmaxed(row, i) = exp(normalized_data(row, 0)) / exp_sum;
				}
			}
			for (unsigned int i = 0; i < targets->cols(); i++)
			{
				softmaxed(round((*targets)(0, i)), i) -= 1;
			}
			return softmaxed;
		}

        std::unique_ptr<Loss> clone() override
        {
            return std::unique_ptr<SparseSoftmaxCrossEntropyLoss>(new SparseSoftmaxCrossEntropyLoss());
        }

	};

}

#endif


