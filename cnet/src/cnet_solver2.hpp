/**
Implementation of common solvers for neural networks.

@file cnet_solver2.hpp
@author Bastian Schoettle EN RC PREC
*/

#ifndef CNET_SOLVER2_HPP
#define CNET_SOLVER2_HPP

#include "cnet_common.hpp"
#include "cnet_loss.hpp"


namespace Cnet
{
	/*
	Abstract base class for Solvers
	*/
	class Solver
	{

	protected:
		float _lr;

	public:

		Solver(const float lr)
		{
			this->_lr = lr;
		}

		void optimize(Graph& graph)
		{
			optimize(graph, this->_lr);
		}

		float get_lr()
		{
			return this->_lr;
		}

		void set_lr(const float lr)
		{
			this->_lr = lr;
		}

		virtual void optimize(Graph& graph, const float lr) = 0;

	};


	class SgdSolver : public Solver
	{
	private:

		std::map<size_t, MatrixRm> _prev_w_upd;
		std::map<size_t, Eigen::VectorXf> _prev_b_upd;

		float _gamma;

	public:

		SgdSolver(const float lr, float gamma) : Solver(lr)
		{
			_gamma = gamma;
		}

		void optimize(Graph& graph, const float lr) override
		{
			MatrixRm* weights = nullptr;
			MatrixRm* gradients = nullptr;
			Eigen::VectorXf* biases = nullptr;
			Eigen::VectorXf* bias_gradients = nullptr;
			for (size_t i = 0; i < graph.get_num_layers(); i++)
			{
				std::shared_ptr<Layer> current_layer = graph.get_layer_by_idx(i);
				weights = current_layer->get_weights();
				gradients = current_layer->get_gradients();
				biases = current_layer->get_bias();
				bias_gradients = current_layer->get_bias_gradients();
				if (this->_prev_w_upd.find(current_layer->get_layer_id()) == this->_prev_w_upd.end())
				{
					MatrixRm mat = MatrixRm::Zero(weights->rows(), weights->cols());
					this->_prev_w_upd[(unsigned long)current_layer->get_layer_id()] = mat;
					Eigen::VectorXf vec = Eigen::VectorXf::Zero(current_layer->get_bias()->rows());
					this->_prev_b_upd[(unsigned long)current_layer->get_layer_id()] = vec;
				}
				MatrixRm* prev_w_upd = &this->_prev_w_upd[(unsigned long)current_layer->get_layer_id()];
				MatrixRm current_w_upd = lr * gradients->array() *  _gamma * prev_w_upd->array();
				(*weights) -= current_w_upd;
				Eigen::VectorXf* prev_b_upd = &this->_prev_b_upd[(unsigned long)current_layer->get_layer_id()];
				MatrixRm current_b_upd = lr * bias_gradients->array() *  _gamma * prev_w_upd->array();
				(*biases) -= current_b_upd;
				*prev_w_upd = current_w_upd;
				*prev_b_upd = current_b_upd;
				current_layer->reset_gradients();
			}
		}
	};

	class AdamSolver : public Solver
	{
	private:
		float _beta_1;
		float _beta_2;
		float _epsilon;
		size_t _internal_step_cnt;

		std::map<size_t, MatrixRm> _v_map_dw;
		std::map<size_t, MatrixRm> _s_map_dw;

		std::map<size_t, Eigen::VectorXf> _v_map_db;
		std::map<size_t, Eigen::VectorXf> _s_map_db;

	public:


		AdamSolver(const float lr, const float beta_1 = 0.9, const float beta_2 = 0.999, const float epsilon = 10e-8) : Solver(lr)
		{
			this->_beta_1 = beta_1;
			this->_beta_2 = beta_2;
			this->_epsilon = epsilon;
			this->_internal_step_cnt = 1;
		}

		void optimize(Graph& graph, const float lr) override
		{
			MatrixRm* weights = nullptr;
			MatrixRm* gradients = nullptr;
			for (size_t i = 0; i < graph.get_num_layers(); i++)
			{
				std::shared_ptr<Layer> current_layer = graph.get_layer_by_idx(i);
				gradients = current_layer->get_gradients();
				weights = current_layer->get_weights();
				if (this->_v_map_dw.find(current_layer->get_layer_id()) == this->_v_map_dw.end())
				{
					MatrixRm vtensor_dw = MatrixRm::Zero(weights->rows(), weights->cols());
					MatrixRm stensor_dw = MatrixRm::Zero(weights->rows(), weights->cols());
					this->_s_map_dw[current_layer->get_layer_id()] = stensor_dw;
					this->_v_map_dw[current_layer->get_layer_id()] = vtensor_dw;
					Eigen::VectorXf vec = Eigen::VectorXf::Zero(current_layer->get_bias()->rows());
					this->_s_map_db[(unsigned long)current_layer->get_layer_id()] = vec;
					this->_v_map_db[(unsigned long)current_layer->get_layer_id()] = vec;
				}
				MatrixRm* stensor_dw = &this->_s_map_dw[(unsigned long)current_layer->get_layer_id()];
				MatrixRm* vtensor_dw = &this->_v_map_dw[(unsigned long)current_layer->get_layer_id()];
				size_t n = weights->rows();
				for (size_t j = 0; j < n; j++)
				{
					size_t o = weights->cols();
					for (size_t k = 0; k < o; k++)
					{
						(*vtensor_dw)(j, k) = this->_beta_1 * (*vtensor_dw)(j, k) + (1.f - this->_beta_1) * (*gradients)(j, k);
						(*stensor_dw)(j, k) = this->_beta_2 * (*stensor_dw)(j, k) + (1.f - this->_beta_2) * pow((*gradients)(j, k), 2.);
						float v_corrected = (*vtensor_dw)(j, k) / (1.f - pow(this->_beta_1, this->_internal_step_cnt));
						float s_corrected = (*stensor_dw)(j, k) / (1.f - pow(this->_beta_2, this->_internal_step_cnt));
						(*weights)(j, k) -= lr * v_corrected / sqrt(s_corrected + this->_epsilon);
						//(*weights)(j, k) -= lr * (*vtensor_dw)(j, k) / sqrt((*stensor_dw)(j, k) + this->_epsilon);
					}
				}
				Eigen::VectorXf* stensor_db = &this->_s_map_db[(unsigned long)current_layer->get_layer_id()];
				Eigen::VectorXf* vtensor_db = &this->_v_map_db[(unsigned long)current_layer->get_layer_id()];
				n = current_layer->get_bias_gradients()->rows();
				for (size_t i = 0; i < n; i++)
				{
					(*vtensor_db)(i) = this->_beta_1 * (*vtensor_db)(i) + (1 - this->_beta_1) * (*current_layer->get_bias_gradients())(i);
					(*stensor_db)(i) = this->_beta_2 * (*stensor_db)(i) + (1 - this->_beta_2) * pow((*current_layer->get_bias_gradients())(i), 2.);
					float v_corrected = (*vtensor_db)(i) / (1 - pow(this->_beta_1, this->_internal_step_cnt));
					float s_corrected = (*stensor_db)(i) / (1 - pow(this->_beta_2, this->_internal_step_cnt));
					(*current_layer->get_bias())(i) -=  lr * v_corrected / sqrt(s_corrected + this->_epsilon);
					//(*current_layer->get_bias())(i) -= lr * (*vtensor_db)(i) / sqrt((*stensor_db)(i) + this->_epsilon);
				}
				current_layer->reset_gradients();
			}
			++this->_internal_step_cnt;
		}
	};

}

#endif


