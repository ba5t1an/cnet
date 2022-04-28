/**
Implementation of several initializers to initialize the layers of a neural network.

@file cnet_initializer.hpp
@author Bastian Schoettle
*/


#ifndef CNET_INITIALIZER_HPP
#define CNET_INITIALIZER_HPP

#include <random>
#include "cnet_common.hpp"

namespace Cnet {

    enum InitializerType { HE_NORMAL = 0, HE_UNIFORM = 1};


    enum IntitializerRule { CONV_INITIALIZER = 0, FC_INITIALIZER = 1};

	/*
	* Base class for initializers
	*/
	class Initializer
	{

	protected:
	    /*
	     * is bias fixed to a value
	     */
		bool _fix_bias;
		float _bias_value;
		bool _use_seed;
		int _seed;
		std::mt19937 _rng;

		void init_rnd_device()
		{
			if (this->_use_seed)
			{
				this->_rng.seed(this->_seed);
			}
			else
			{
				this->_rng.seed(std::random_device()());
			}
		}

	public:

		Initializer() {
			this->_fix_bias = true;
			this->_bias_value = 0.f;
			this->_use_seed = true;
			this->_seed = 1;
			init_rnd_device();
		}

		Initializer(bool fix_bias, bool use_seed, float bias_value = 0.01, int seed = 0) {
			this->_fix_bias = fix_bias;
			this->_bias_value = bias_value;
			this->_use_seed = use_seed;
			this->_seed = seed;
			init_rnd_device();
		}

		void set_bias(float bias) {
			this->_fix_bias = true;
			this->_bias_value = bias;
		}

		void set_seed(int seed) {
			this->_use_seed = true;
			this->_seed = seed;
		}

        static inline std::shared_ptr<Initializer> create_initializer(InitializerType type);

		static inline std::shared_ptr<Initializer> get_initializer_ptr(const InitializerType type);

		virtual void initialize(MatrixRm& weights, Eigen::VectorXf& biases, IntitializerRule type = FC_INITIALIZER) = 0;
	
	protected:
		
		float draw_truncated_normal(std::normal_distribution<float>& dist) {
			float random_num = dist(this->_rng);
			while (random_num < dist.mean() - dist.stddev() && random_num > dist.mean() + dist.stddev())
			{
				random_num = dist(this->_rng);
			}
			return random_num;
		}

		float draw_normal(std::normal_distribution<float>& dist) {
			return dist(this->_rng);
		}

	};

	class HeUniformInitializer : public Initializer
	{

	public:

		void initialize(MatrixRm& weights, Eigen::VectorXf& biases, IntitializerRule type = FC_INITIALIZER) override
		{
			float k = 0.f;
			if (type == CONV_INITIALIZER)
			{
				k = sqrt(2.f / (weights.rows() * weights.cols()));
			}
			else
			{
				k = sqrt(2.f / (weights.cols()));
			}
			std::uniform_real_distribution<float> dist(-k, k);
			if (this->_fix_bias)
			{
				for(unsigned int i = 0; i < biases.rows(); ++i){
					biases(i,0) = dist(_rng);
				}

			}
			else
			{
				for(unsigned int i = 0; i < biases.rows(); ++i){
					biases(i,0) = dist(_rng);
				}

			}
			
			for(unsigned int i = 0; i < weights.rows(); ++i){
				for(unsigned int j = 0; j < weights.cols(); ++j)
				{
					weights(i,j) = dist(_rng);
				}
			}

		}
	};

	class HeNormalInitializer : public Initializer
	{

	public:

		void initialize(MatrixRm& weights, Eigen::VectorXf& biases, IntitializerRule type = FC_INITIALIZER) override
		{
			float k;
			if (type == CONV_INITIALIZER)
			{
				k = sqrt(2.f / (weights.rows() * weights.cols()));
			}
			else
			{
				k = sqrt(2.f / (weights.cols()));
			}

			std::normal_distribution<float> dist(0., k);
			if (this->_fix_bias)
			{
				for (unsigned int i = 0; i < biases.rows(); ++i) {
					biases(i, 0) = this->draw_truncated_normal(dist);
				}

			}
			else
			{
				for (unsigned int i = 0; i < biases.rows(); ++i) {
					biases(i, 0) = this->draw_truncated_normal(dist);
				}

			}

			for (unsigned int i = 0; i < weights.rows(); ++i) {
				for (unsigned int j = 0; j < weights.cols(); ++j)
				{
					weights(i, j) = this->draw_truncated_normal(dist);
				}
			}

		}
	};

        /*
    * Factory method to create shared_ptr of weight initializer.
    */
    std::shared_ptr<Initializer> Initializer::create_initializer(InitializerType type)
    {
        if(type == HE_UNIFORM)
        {
            return std::move(std::make_shared<HeUniformInitializer>());
        }
        return std::move(std::make_shared<HeNormalInitializer>());
    }

    /*
    * Map to store initializers -> One ptr per type is enough for initial creation
    * Same here, ugly until C++17...inline variables
    */
    static std::map <InitializerType, std::shared_ptr<Initializer> > _initializer_ptr_map;

    std::shared_ptr<Initializer> Initializer::get_initializer_ptr(const InitializerType type)
    {
        if (_initializer_ptr_map.find(type) != _initializer_ptr_map.end())
        {
            return _initializer_ptr_map[type];
        }
        _initializer_ptr_map[type] = Initializer::create_initializer(type);
        return _initializer_ptr_map[type];
    }


}
#endif
