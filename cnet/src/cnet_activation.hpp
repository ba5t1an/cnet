/**
Implementation of neural network activation functions.

@file cnet_activation.hpp
@author Bastian Schoettle
*/


#ifndef CNET_ACTIVTION_HPP
#define CNET_ACTIVTION_HPP

#include <math.h>
#include <memory>
#include "cnet_common.hpp"
#include "cnet_layer.hpp"

#ifdef CNET_DEBUG
#include <experimental/filesystem>
#endif // CNET_DEBUG

namespace Cnet
{

	enum ActivationType {
		RELU = 0,
		LEAKY_RELU = 1,
		LOGISTIC = 2,
		LINEAR = 3,
		IDENTITY = 4,
		SOFTMAX = 5,
		LECUN_TANH = 6,
		TANH = 7
	};


    class Activation {

    protected:

		ActivationType _activation_type;

	public:

		ActivationType get_type() {
			return this->_activation_type;
		}

		virtual float forward(float x) = 0;
		virtual float backward(float x) = 0;

		static inline std::shared_ptr<Activation> create_activation(ActivationType type);

        /*
        * Function to retrive already created instances of activations from a map.
        */
        static inline std::shared_ptr<Activation> get_activation_ptr(const ActivationType type);


        virtual MatrixRm forward(MatrixRm& data)
		{
			MatrixRm result = data.unaryExpr([&](float x) -> float {
				return this->forward(x);
			});
			return result;
		}

		virtual MatrixRm backward(MatrixRm& data)
		{
			MatrixRm result = data.unaryExpr([this](float x) -> float {
				return this->backward(x);
			});
			return result;
		}

	};

	class ReLU : public Activation {
	public:
		ReLU() {
			this->_activation_type = RELU;
		}

		float forward(float x) override {
			if (x <= 0.f)
			{
				return 0.f;
			}
			return x;
		}

		float backward(float x) override {
			if (x <= 0.f)
			{
				return 0.f;
			}
			return 1.f;
		}

	};

	class Tanh : public Activation {
	public:
		Tanh() {
			this->_activation_type = TANH;
		}

		float forward(float x) override {
			//std::cout << x << std::endl;
			return tanh(x);//(exp(x) - exp(-x)) / (exp(x) + exp(-x));
		}

		float backward(float x) override {
			float th = forward(x);
			return 1.f - th*th;
		}

	};

	class LeakyReLU : public Activation {
	public:
		LeakyReLU() {
			this->_activation_type = LEAKY_RELU;
		}
		float forward(float x) override {
			if (x > 0.f)
			{
				return x;
			}
			return 0.01f*x;
		}

		float backward(float x) override {
			if (x > 0.f)
			{
				return 1.f;
			}
			return 0.01f*x;
		}
	};

	class LeCunTanh: public Activation {
	public:
		LeCunTanh() {
			this->_activation_type = LECUN_TANH;
		}

		float forward(float x) override {
			if (x > 0.f)
			{
				return x;
			}
			return 0.01f*x;
		}

		float backward(float x) override {
			if (x > 0.f)
			{
				return 1.f;
			}
			return 0.01f*x;
		}
	};

	class Logistic : public Activation {

	public:
		Logistic() {
			this->_activation_type = LOGISTIC;
		}

		float forward(float x) override {
			return (1.f / (1.f + exp(-x)));
		}

		float backward(float x) override {
			return this->forward(x)*(1.f - this->forward(x));
		}
	};

	class Linear : public Activation {
		Linear() {
			this->_activation_type = LINEAR;
		}
	private:
		float _c;

		Linear(float c) {
			this->_c = c;
		}
	public:
		float forward(float x) override {
			return this->_c * x;
		}

		float backward(float x) override {
			return this->_c+(0*x); //Useless multipy by 0 to avoid compiler warning...
		}
	};

	class Identity : public Activation {

	public:
		Identity() {
			this->_activation_type = IDENTITY;
		}
		float forward(float x) override {
			return x;
		}

		float backward(float x) override {
			return 1+(x*0); //Useless multiply with zero to avoid compiler warning...
		}
	};


	class Softmax : public Activation {
	private:

		MatrixRm _softmax_cache;
		MatrixRm _data_cache;

		inline MatrixRm softmax(MatrixRm* data) {
			size_t m = data->cols();
			MatrixRm normalized_data = data->array() - data->maxCoeff();
			MatrixRm softmax_out(1, m);

			float exp_sum = 0.f;
			for (size_t j = 0; j < m; ++j)
			{
				exp_sum += exp(normalized_data(0, j));
			}
			for (size_t j = 0; j < m; ++j)
			{
				softmax_out(0, j) = exp(normalized_data(0, j)) / exp_sum;
			}

			return softmax_out;
		}

		inline void softmax_sparse(MatrixRm& data, MatrixRm& out) {
			MatrixRm chunk(data.rows(), 1);
			MatrixRm sm_chunk(data.rows(), 1);
			for (unsigned int i = 0; i < data.cols(); ++i)
			{
				MatrixRm chunk = data.block(0, i, data.rows(), 1).transpose();
				out.block(0, i, data.rows(), 1) = softmax(&chunk).transpose();
			}
		}

	public:
		Softmax() {
			this->_activation_type = SOFTMAX;
			this->_softmax_cache = MatrixRm();
			this->_data_cache = MatrixRm();
		}


		float forward(float x) override {
			throw std::runtime_error("Operation not supported for v = " + std::to_string(x));
		}

		float backward(float x) override {

			for (unsigned int i = 0; i < this->_softmax_cache.cols(); ++i)
			{
				if (this->_data_cache(0, i) == x)
				{
					return this->_softmax_cache(0, i) * (1 - this->_softmax_cache(0, i));
				}
			}
			return 0.f;
		}

		MatrixRm backward(MatrixRm& data) override
		{
			MatrixRm sm_values = softmax(&data);
			MatrixRm result = sm_values.unaryExpr([=](float x) -> float {
				return (x * (1.f - x));
			});
			return result;
		}

		MatrixRm forward(MatrixRm& data) override
		{
			if (data.rows() > 1)
			{
				softmax_sparse(data, _softmax_cache);
			}
			else
			{
				this->_softmax_cache = softmax(&data);
			}
			this->_data_cache = data;
			return this->_softmax_cache;
		}
	};

	/*
	* Factory method to create shared_ptr of activation functions.
	*/
	std::shared_ptr<Activation> Activation::create_activation(ActivationType type)
    {
		if (type == LEAKY_RELU)
		{
            return std::move(std::make_shared<LeakyReLU>());
		}
		if (type == SOFTMAX)
		{
            return std::move(std::make_shared<Softmax>());
		}
		if (type == LOGISTIC)
		{
            return std::move(std::make_shared<Logistic>());
		}
		if (type == TANH)
		{
            return std::move(std::make_shared<Tanh>());
		}
        return std::move(std::make_shared<ReLU>());
	}

	/*
    *  Map to store activations -> One ptr per type is enough for initial creation
    *  Stays ugly until C++17... your choice of the compiler version...
    */
    static std::map <ActivationType, std::shared_ptr<Activation>> _activation_ptr_map;



    std::shared_ptr<Activation> Activation::get_activation_ptr(const ActivationType type)
    {
        if (_activation_ptr_map.find(type) != _activation_ptr_map.end())
        {
            return _activation_ptr_map[type];
        }
        _activation_ptr_map[type] = Activation::create_activation(type);
        return _activation_ptr_map[type];
    }

}

#endif
