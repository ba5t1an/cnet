/* Copyright (C) Siemens Logistics GmbH 2019 */

/**
Implementation of several neural network layers based on Eigen

@file cnet_layer.hpp
@author Bastian Schoettle

TODO:
	-	Check if setting output to zero before fwd is really neccessary.

*/

#ifndef CNET_LAYER_HPP
#define CNET_LAYER_HPP


#include <vector>
#include <memory>
#include <string>
#include <map>

#include "cnet_common.hpp"
#include "cnet_activation.hpp"
#include "cnet_initializer.hpp"
#include "cnet_visitor.hpp"


namespace Cnet
{
	/*
	* LayerType enum to store different layer types
	*/
	enum LayerType
	{
		INPUT_LAYER = 0,
		CONV2D_LAYER = 1,
		FC_LAYER = 2,
		MAXPOOL2D_LAYER = 3,
		CONV2D_T_LAYER = 4,
		CONV2D_CC_LAYER = 5,
		DROP_LAYER = 6
	};


	/*
	* Padding policy for convolution layer
	*/
	enum PaddingPolicy {
		VALID = 0, SAME = 1
	};


	/*
	* Mapping between LayerType and associated string
	*/
	const std::map<LayerType, std::string> LAYER_MAP = { { INPUT_LAYER, "InputLayer" },{ CONV2D_LAYER, "Conv2dLayer" },{ FC_LAYER, "FcLayer" },{ MAXPOOL2D_LAYER, "Maxpool2dLayer" },{ CONV2D_T_LAYER, "Conv2dTransposeLayer" },{ CONV2D_CC_LAYER, "CropAndConcatConv2DLayer" } };

	/*
	* Class defintion for base layer
	*
	* Any Layer will inherit the methods and members of this class. It's abstract,
	* hence extending layers must implement forward() and backward(...).
	*/

	class Layer
	{

	protected:

		/*
		* Layer type to identify layer
		*/
		LayerType _layer_type;

		/*
		* Layer name, used mainly for debug purpose
		*/
		std::string _layer_name;

		/*
		* Layer's id across application
		*/
		int _layer_id;

		/*
		* Member to verify that this layer has an input layer
		*/
		bool _has_previous;

		/*
		* Pointer to the previous layer of this layer
		*/
		std::shared_ptr<Layer> _previous_layer;

		/*
		* Activation function for this layer
		*/
		std::shared_ptr<Activation> _activation;

		/*
		* Weight initializer of this layer
		*/
		std::shared_ptr<Initializer> _initializer;

		/*
		* Size of an output channel
		*/
		size_t _output_size;

		/*
		* Number of output channels
		*/
		size_t _output_channels;

		/*
		* Weight MatrixRm to store weights associated with the layer.
		* Dimensions n*n*m.
		*/
		MatrixRm _weights;

		/*
		* Gradient matrix to store the gradients of the weights
		* associated with the layer. MatrixRm dimensions n*n*m.
		*/
		MatrixRm _gradients;

		/*
		* Biases associated with the layer's weights. This is actually a vector represented by a 1xn matrix.
		*/
		Eigen::VectorXf _biases;

		/*
		* Gradients for the biases. Datastractur is identical to _biases'.
		*/
		Eigen::VectorXf _bias_gradients;

		/*
		* Output of the layer. This matrix is retrieved by the next layer in the graph.
		*/
		MatrixRm _output;

		/*
		* Deltas for the output nodes in case of FC layer.
		* Otherwise deltas to the input nodes of the current layer.
		*/
		MatrixRm _dx;

        /*
        * Deltas for the output nodes in case of FC layer.
        * Otherwise deltas to the input nodes of the current layer.
        */
        MatrixRm _skip_dx;

		/*
		* Cache to store dot products before activation is applied.
		*/
		MatrixRm _cache;

		/*
		 * Skip layer such as concat or residual. Layer is used to propagate gradients
		 */
        std::shared_ptr<Layer> _bwd_skip_connection;

        /*
         * Skip layer such as concat or residual. Layer is used to propagate another layer's x to this layer
         */
        std::shared_ptr<Layer> _fwd_skip_connection;

		/*
		* Method to initialize layer's parameters (bias and weights). If no initializer is provided, the current default (Xavier)
		* will be used for initialization.
		*/
		void initialize_parameters()
		{
			IntitializerRule type = CONV_INITIALIZER;
			if (this->_layer_type == FC_LAYER)
			{
				type = FC_INITIALIZER;
			}
			if (_initializer)
			{
				_initializer->initialize(_weights, _biases, type);
			}
			else
			{
                _initializer = Initializer::get_initializer_ptr(HE_NORMAL);
                _initializer->initialize(_weights, _biases, type);
			}
		}


        void merge_deltas(MatrixRm* deltas)
        {
		    if(_bwd_skip_connection)
            {
				deltas->array() +=  _bwd_skip_connection->get_skip_dx()->array();
            }
        }


		/*
		* init() is used to precompute certain numbers and to set the shape of all necessary data structures.
		*/
		virtual void init() = 0;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor for Layer base class
			*/
			Layer(std::string layer_name = "BaseLayer") {
			_layer_name = layer_name;
			_has_previous = false;
			_activation = nullptr;
			_activation = Activation::get_activation_ptr(RELU);
			_layer_id = -1;
		}

		/*
		* Constructor for Layer base class. Overloaded.
		*/
		Layer(std::shared_ptr<Activation > activation, std::string layer_name = "BaseLayer") {
			_layer_name = layer_name;
			_has_previous = false;
			_activation = nullptr;
			if (activation == nullptr)
			{
				_activation = Activation::get_activation_ptr(RELU);
			}
			else
			{
				_activation = activation;
			}
			_layer_id = -1;
		}

		/*
		* Constructor for Layer base class. Overloaded.
		*/
		Layer(std::shared_ptr<Activation > activation, std::shared_ptr<Initializer> initializer, std::string layer_name = "BaseLayer") {
			_layer_name = layer_name;
			_has_previous = false;
			_activation = nullptr;
			if (activation == nullptr)
			{
				_activation = Activation::get_activation_ptr(RELU);
			}
			else
			{
				_activation = activation;
			}
            _initializer = initializer;
			_layer_id = -1;
		}

		/*
		* Constructor for Layer base class. Overloaded.
		*/
		Layer(std::shared_ptr<Layer >& previous_layer, std::shared_ptr<Activation> activation = nullptr, std::shared_ptr<Initializer> initalizer = nullptr, std::string layer_name = "BaseLayer") {
			_layer_name = layer_name;
			_previous_layer = previous_layer;
			_activation = nullptr;
			if (activation == nullptr)
			{
				_activation = Activation::get_activation_ptr(RELU);
			}
			else
			{
				_activation = activation;
			}
			_has_previous = true;
            _initializer = initalizer;
			_layer_id = -1;
		}


		/*
		* Constructor for Layer base class. Overloaded.
		*/
		Layer(std::shared_ptr<Layer>& previous_layer, std::string layer_name = "BaseLayer") {
			_layer_name = layer_name;
			_previous_layer = previous_layer;
			_activation = nullptr;
			_has_previous = true;
			_layer_id = -1;
		}

		/*
		* Returns current layer's activation
		*/
		std::shared_ptr<Activation> get_activation() {
			return _activation;
		}

		std::string get_layer_name()
		{
			return _layer_name;
		}

		std::string get_layer_type_as_string()
		{
			return LAYER_MAP.at(get_layer_type());
		}

		/*
		* Setter for layer id
		*/
		void set_layer_id(int layer_id) {
			_layer_id = layer_id;
		}

		/*
		* Returns layer's type
		*/
		LayerType get_layer_type() {
			return _layer_type;
		}

		/*
		* Returns layer's id
		*/
		int get_layer_id() {
			return _layer_id;
		}

		/*
		* Resets layer's gradients (bias and weights)
		*/
		virtual void reset_gradients() {
			if (_layer_type != INPUT_LAYER)
			{
				_gradients.setZero();
				_bias_gradients.setZero();
			}
		}

		/*
		* Resets layer's deltas
		*/
		void reset_dx() {
			_dx.setZero();
		}

		/*
		* Setter for previous layer.
		* NOTE: Will call layer specific init() method, so make sure that init() is overriden.
		*/
		virtual void set_previous_layer(std::shared_ptr<Layer> previous_layer)
		{
			_previous_layer = previous_layer;
			init();
		}

		/*
		* Reset layer's deltas and gradients
		*/
		virtual void reset_layer() {
			reset_gradients();
			reset_dx();
		}

		/*
		* Returns true if current layer has a previous one
		*/
		bool has_previous() {
			return _has_previous;
		}

		/*
		* Returns the previous layer
		*/
		std::shared_ptr<Layer >& get_previous_layer() {
			return _previous_layer;
		}

		/*
		Returns pointer to delta matrix
		*/
		MatrixRm* get_dx() {
			return &_dx;
		}

		/*
		Returns layer's output
		*/
		virtual MatrixRm* get_output() {
			return &_output;
		}

		/*
		Returns layer's cache
		*/
		MatrixRm* get_cache() {
			return &_cache;
		}

		/*
		Returns layer's weights
		*/
		MatrixRm* get_weights() {
			return &_weights;
		}

		/*
		Returns bias' of current layer
		*/
		Eigen::VectorXf* get_bias() {
			return &_biases;
		}

		/*
		Returns bias gradients of current layer
		*/
		Eigen::VectorXf* get_bias_gradients() {
			return &_bias_gradients;
		}

		/*
		Returns weight gradients of current layer
		*/
		MatrixRm* get_gradients() {
			return &_gradients;
		}

		MatrixRm* get_skip_dx()
        {
		    return &_skip_dx;
        }

		std::shared_ptr<Layer> get_skip_layer()
		{
			return _bwd_skip_connection;
		}

		/*
		Accept visitor here
		*/
		void accept(Visitor* visitor)
		{
			visitor->visit(this);
		}

		size_t get_output_size()
		{
			return _output_size;
		}

		size_t get_output_channels()
		{
			return _output_channels;
		}


		/*
		* Pure virtual method for the forward pass which must be implemented depending on layer type.
		*/
		virtual void forward(bool training = false) = 0;

		/*
		* NOTE: Propagates input to the top layer using parent node.
		*/
		virtual void forward(MatrixRm* data)
		{
			_output = *data;
		}

		/*
		* Purly virtual method wich must be implemented by different layer types. It's used to perform backpropagation
		* of the specific layer.
		*/
		virtual void backward(MatrixRm* deltas) = 0;


        void set_bwd_skip(std::shared_ptr<Layer>& skip_connection)
        {
            _bwd_skip_connection = skip_connection;
        }
	};

	/*
	* Class defintion for input layer
	*/
	class InputLayer : public Layer
	{
	private:

		MatrixRm* _output_ptr;

		void init() override
		{
		}

	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor for Input layer. This layer must not have a previous layers.
			*/
			InputLayer(const size_t in_features, const size_t in_channels, std::string layer_name = "InputLayer") : Layer(layer_name) {
			_layer_type = INPUT_LAYER;
			_output_channels = in_channels;
			_output_size = (in_features * in_features);
			init();
		}

		/*
		* Forward method for input layer. NOTE: This will throw an exception since forwarding
		* nothing through the consecutive layers will make no sense and probably cause runtime exceptions.
		* So, simply dont't use...
		*/
		void forward(bool training = false) override {
            (void) training;
			throw std::runtime_error("Operation not supported");
		}

		void forward(MatrixRm* data) override
		{
			_output_ptr = data;
		}

		MatrixRm* get_output() override
		{
			return _output_ptr;
		}


		/*
		* Backwards operation for InputLayer is kinda useless.
		* Hence, an exception will be thrown if someone tries to call it.
		* The method is still implemented to fullfill the requirements of the interface.
		*/
		void backward(MatrixRm* deltas) override
		{
			(void)deltas;
			throw std::runtime_error("Operation not supported");
		}

        static inline std::shared_ptr<InputLayer> create(const size_t width, const size_t num_channels)
        {
            return std::move(std::make_shared<InputLayer>(width, num_channels));
        }

	};

	/*
	* Class defintion for fully-connected layer.
	* This is implemented use the following memory layout:
	*
	* Input:				R^2 will be flattend to R^1.
	* Weights, Gradients:	R^2, the matrix will have one row for each node of this layer and
	*						one column for each incoming node.
	* Deltas:				R^1 or R^2, depending of the incoming nodes. Deltas will automatically expanded to
	*						match the previous layer's expectations.
	*/
	class FcLayer : public Layer
	{
	private:

		/*
		* Number of input features.
		*/
		size_t _in_size;

		/*
		* If true, input was flattend and will be expanded during backprop.
		*/
		bool _is_flattend;

		/*
		* Matrix to hold flattend data if _is_flattend is true.
		*/
		MatrixRm _flattend_data;

		/*
		* Initilializer method to setup data structurs needed by the layer. It also precomputes numbers needed during forward
		* and backprop.
		*/
		void init()
		{
			_in_size = _previous_layer->get_output_size() * _previous_layer->get_output_channels();
			_output = MatrixRm::Zero(1, _output_size);
			_cache = MatrixRm::Zero(1, _output_size);
			_weights = MatrixRm::Zero(_output_size, _in_size);
			_gradients = MatrixRm::Zero(_output_size, _in_size);
			_biases = Eigen::VectorXf::Zero(_output_size);
			_bias_gradients = Eigen::VectorXf::Zero(_output_size);
			_dx = MatrixRm::Zero(1, _in_size);
			if (_previous_layer->get_output()->rows() > 1)
			{
				_is_flattend = true;
				_flattend_data.resize(1, _in_size);
				_flattend_data.setZero();
			}
			else
			{
				_is_flattend = false;
			}
			initialize_parameters();
		}

		/*
		* Backwards method of this layer. It performs the computation of the weight gradients and deltas for the above layer.
		* NOTE: This layer will only partially compute the deltas for the previous layer. This is because the layer's activation
		* is not sufficiently available here. Hence, the previous layer has to apply the derivative of it's activation to deltas by itself.
		*/
		void fc_bwd(MatrixRm* input, MatrixRm* deltas)
		{
			size_t n = _output.cols();
			size_t m = input->cols();
			for (size_t i = 0; i < n; i++)
			{
				(*deltas)(0, i) *= _activation->backward(_cache(0, i));
				for (size_t j = 0; j < m; j++)
				{
					_dx(0, j) += _weights(i, j) * (*deltas)(0, i);
					_gradients(i, j) += (*deltas)(0, i) * (*input)(0, j);
				}
				_bias_gradients(i) += (*deltas)(0, i); // bias
			}
			if (_is_flattend)
			{
				_dx = MatrixRmMap(_dx.data(), _previous_layer->get_output()->rows(), _previous_layer->get_output()->cols());
			}
		}

		/*
		* Will perform the identical procedure as forward() but this time using matrix multiplications.
		* NOTE: Currently, the bias is probably added inefficiently
		*/
		void fc_fwd_gemm(MatrixRm* input)
		{
			MatrixRm temp = _weights * input->transpose();
			_cache = Eigen::Map<MatrixRm>(temp.data(), 1, temp.rows());
			_cache.row(0).array() += _biases.array();
			_output = _activation->forward(_cache);
		}

	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* First constructor for FcLayer
			*/
			FcLayer(std::shared_ptr<Layer>& previous_layer, const size_t out, std::shared_ptr<Activation> activation = nullptr, std::shared_ptr<Initializer> initializer = nullptr, std::string layer_name = "FcLayer") : Layer(previous_layer, activation, initializer, layer_name)
		{
			_layer_type = FC_LAYER;
			_output_size = out;
			_output_channels = 1;
			init();
		}

		size_t get_num_inputs()
		{
			return _in_size;
		}

		/*
		* Public forward method for FcLayer. Currently, forward_gemm() is used. Hence, a matrix multiplication is performed.
		* This one is called during inference and training.
		*/
		void forward(bool training = false) override
		{
            (void) training;
			_cache.setZero();
			if (_is_flattend)
			{
				MatrixRm reshaped = MatrixRmMap(_previous_layer->get_output()->data(), 1, _previous_layer->get_output()->size());
				fc_fwd_gemm(&reshaped);
			}
			else
			{
				fc_fwd_gemm(_previous_layer->get_output());
			}
		}
		/*
		* Public backwards method. This is also called during inference and training.
		*/
		void backward(MatrixRm* deltas) override
		{
			_dx.setZero();
			if (_is_flattend)
			{
				_dx = MatrixRmMap(_dx.data(), 1, _dx.size());
				MatrixRm reshaped = MatrixRmMap(_previous_layer->get_output()->data(), 1, _previous_layer->get_output()->size());
				fc_bwd(&reshaped, deltas);
			}
			else
			{
				fc_bwd(_previous_layer->get_output(), deltas);
			}
		}

        static inline std::shared_ptr<Layer> create(std::shared_ptr<Layer> previous_layer, const size_t out, ActivationType activation = RELU, InitializerType initializer = HE_NORMAL)
        {
            return std::move(std::make_shared<FcLayer>(previous_layer, out, Activation::get_activation_ptr(activation), Initializer::get_initializer_ptr(initializer)));
        }

	};

	/*
	*
	* Class defintion for 2d convolutional layer, but not actual convolution is performed. It's actually a cross-correlation.
	* The memory layout is the following:
	* Input:				R^2 here, the input is a matrix where every row is the channel of an image or a feature map. An NxN single channel image
	*						is here represented as 1xN*N matrix.
	* Weights, Gradients:	The weights and associated gradients are a bit weird but serve the purpose of performaing the cross-correlation as GEMM.
	Hence, it's a NxM Matrix where N is the number of filters and M a C*K*K. Where C is the number of input channels and K*K
	the kernel size, e.g. 3x3 = 9.
	*
	*/
	class Conv2dLayer : public Layer
	{
	private:

		size_t _padded_width;
		/*
		* Init method for Conv2DLayer. It's used to precompute certain relevant numbers and sets the shapes of
		* weights, biases, outputs and so on. It also calls initialize_parameters to initialize the weights with
		* rendom numbers from a specific distribution.
		*/
		void init()
		{
			_num_channels = _previous_layer->get_output_channels();
			_input_width = (size_t)sqrt(_previous_layer->get_output_size());
			_output_width = (size_t)floor((_input_width - _conv_params.kernel_width() + _conv_params.padding() ) / _conv_params.stride()) + 1;
			_output_size = (unsigned int)pow(_output_width, 2);
			_output_channels = _num_kernels;
			if (_conv_params.padding() > 0)
			{
				_padded_width = _input_width + _conv_params.padding();
				_padded_input = MatrixRm::Zero(_num_channels, (unsigned int)pow(_padded_width, 2.));
			}
			_weights = MatrixRm::Zero(_num_kernels, _num_channels*_conv_params.kernel_size());
			_gradients = MatrixRm::Zero(_num_kernels, _num_channels*_conv_params.kernel_size());
			_output = MatrixRm::Zero(_num_kernels, _output_size);
			_cache = MatrixRm::Zero(_num_kernels, _output_size);
			_dx = MatrixRm::Zero(_num_channels, _previous_layer->get_output_size());
			_biases = Eigen::VectorXf::Zero(_num_kernels);
			_bias_gradients = Eigen::VectorXf::Zero(_num_kernels);
			_im2col_input = MatrixRm::Zero(_num_channels*_conv_params.kernel_size(), _output_width);
			initialize_parameters();
		}

	protected:
		size_t _num_channels;
		size_t _input_width;
		size_t _num_kernels;
		size_t _output_width;
		MatrixRm _padded_input;

		MatrixRm _reshaped_output;

		ConvolutionParams _conv_params = {3,1,0};
		/*
		Keep im2col transformed input for bwd pass
		*/
		MatrixRm _im2col_input;

		/*
		* Performs cross-correlation as matrix multiplication between the previous layer's output and a given set of filters.
		* NOTE: This is really no convolution...no kernel flipping and so on. If you want a convolution, just flip the kernels.
		* But probably copy the method and give it a different name.
		*/
		inline void correlate2d_fwd_gemm(MatrixRm* input) {
			_im2col_input.setZero();
			im2col(input, &_im2col_input, _conv_params);
			MatrixRm result = _weights * _im2col_input;
			_cache = MatrixRmMap(result.data(), _num_kernels, _output_width*_output_width);
			for (size_t i = 0; i < _num_kernels; i++)
			{
				_cache.row(i).array() += _biases(i, 0);
			}
			_output = _activation->forward(_cache);
		}


		inline void correlate2d_bwd_gemm(MatrixRm* input, MatrixRm* dout)
		{
			dout->array() *= _activation->backward(_cache).array();
			const size_t row_len = (size_t)(dout->cols() * dout->rows()) / _weights.rows();
			MatrixRm dout_rs = MatrixRmMap(dout->transpose().data(), _weights.rows(), row_len);
			MatrixRm wgrad_tmp = dout_rs * _im2col_input.transpose();
			_gradients.array() += MatrixRmMap(wgrad_tmp.data(), _weights.rows(), _weights.cols()).array();
			if (_previous_layer->get_layer_type() != INPUT_LAYER)
			{
				MatrixRm dx_col = _weights.transpose() * dout_rs;
				col2im(input, &dx_col, &_dx, _conv_params);
			}
			for (size_t i = 0; i < _num_kernels; ++i)
			{
				_bias_gradients(i, 0) = dout->row(i).sum();
			}
		}


	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor for Conv2dLayer
			*/
			Conv2dLayer(std::shared_ptr<Layer>& previous_layer, const size_t out, const size_t kernel_width, size_t stride, PaddingPolicy padding = VALID, std::shared_ptr<Activation> activation = nullptr, std::shared_ptr<Initializer> initializer = nullptr, std::string layer_name = "Conv2dLayer") : Layer(previous_layer, activation, initializer, layer_name)
		{
			_layer_type = CONV2D_LAYER;
			size_t pad = 0;
			if (padding == SAME)
			{
				pad = (kernel_width - 1) / stride;
			}
			_conv_params.kernel_width(kernel_width);
			_conv_params.padding(pad);
			_conv_params.stride(stride);
			_num_kernels = out;
			init();
		}

		size_t get_num_kernels()
		{
			return _num_kernels;
		}

		/*
		* Getter for stride.
		*/
		ConvolutionParams get_conv_params()
		{
			return _conv_params;
		}

		MatrixRm* get_output() override
		{
			if (_output.cols() == 1 && _output.rows() > 1)
			{
				_reshaped_output = MatrixRmMap(_output.data(), 1, _output.rows());
				return &_reshaped_output;
			}
			return &_output;
		}

		/*
		* Performs forward pass of conv2d layer.
		* TODO: If possible, try to avoid looping over rows for adding biases. This also applies to the GEMM version of cc.
		*/
		void forward(bool training = false) override
		{
			(void)training;
			if (_conv_params.padding() > 0)
			{
				for (unsigned int channel = 0; channel < _padded_input.rows(); ++channel)
				{
					MatrixRmMap(_padded_input.row(channel).data(), _padded_width, _padded_width).block(_conv_params.padding(), _conv_params.padding(), _input_width, _input_width)
						= MatrixRmMap(get_previous_layer()->get_output()->row(channel).data(), _input_width, _input_width);
				}
				correlate2d_fwd_gemm(&_padded_input);
			}
			else
			{
				correlate2d_fwd_gemm(get_previous_layer()->get_output());
			}
		}

		/*
		* Performs backward pass of Conv2dlayer
		*/
		void backward(MatrixRm* deltas) override
		{
			if (deltas->rows() == 1 && deltas->cols() > 1)
			{
				deltas->resize(deltas->cols(), 1);
			}
			_dx.setZero();
			merge_deltas(deltas);
			if (_conv_params.padding() > 0)
			{
				correlate2d_bwd_gemm(&_padded_input, deltas);
			}
			else
			{
				correlate2d_bwd_gemm(_previous_layer->get_output(), deltas);
			}
		}

        static inline std::shared_ptr<Layer> create(std::shared_ptr<Layer> previous_layer, const size_t num_kernels, const size_t kernel_width, const size_t stride, PaddingPolicy padding = VALID, ActivationType activation = RELU, InitializerType initializer = HE_NORMAL)
        {
            return std::make_shared<Conv2dLayer>(previous_layer, num_kernels, kernel_width, stride, padding, Activation::get_activation_ptr(activation), Initializer::get_initializer_ptr(initializer));
        }

	};


	/*
	*
	* Class defintion for 2d convolutional layer, but not actual convolution is performed. It's actually a cross-correlation.
	* The memory layout is the following:
	* Input:				R^2 here, the input is a matrix where every row is the channel of an image or a feature map. An NxN single channel image
	*						is here represented as 1xN*N matrix.
	* Weights, Gradients:	The weights and associated gradients are a bit weird but serve the purpose of performaing the cross-correlation as GEMM.
	Hence, it's a NxM Matrix where N is the number of filters and M a C*K*K. Where C is the number of input channels and K*K
	the kernel size, e.g. 3x3 = 9.
	*
	*/
	class Conv2dTransposeLayer : public Conv2dLayer
	{
	private:
		/*
		* Init method for Conv2DTransposeLayer. It's used to precompute certain relevant numbers and sets the shapes of
		* weights, biases, outputs and so on. It also calls initialize_parameters to initialize the weights with
		* random numbers from a specific distribution.
		*/
		void init()
		{
			_num_channels = _previous_layer->get_output_channels();
			_input_width = (size_t) sqrt(_previous_layer->get_output_size());
			_output_width = _conv_params.stride() * (_input_width - 1) + _conv_params.kernel_width();
			_output_size = (size_t) pow(_output_width, 2.);
			_output_channels = _num_kernels;
			_weights = MatrixRm::Zero(_num_kernels, _conv_params.kernel_size()*_num_channels);
			_gradients = MatrixRm::Zero(_num_kernels, _conv_params.kernel_size()*_num_channels);
			_dx = MatrixRm::Zero(_num_channels, _previous_layer->get_output_size());
			_biases = Eigen::VectorXf::Zero(_num_kernels);
			_bias_gradients = Eigen::VectorXf::Zero(_num_kernels);
			_output = MatrixRm::Zero(_num_kernels, _output_width * _output_width);
			_cache = MatrixRm::Zero(_num_kernels, _output_width* _output_width);
			initialize_parameters();
		}

		inline void correlate2d_t_fwd(MatrixRm* input) {
			_cache.setZero();
			const size_t input_depth = input->rows();
			const size_t kernel_moves = (size_t)((sqrt(_output.cols()) - _conv_params.kernel_width()) / _conv_params.stride()) + 1;
			for (unsigned int current_kernel = 0; current_kernel < _weights.rows(); current_kernel++)
			{
				MatrixRm kernel_weight = Eigen::Map<MatrixRm>(_weights.row(current_kernel).data(), _num_channels, _conv_params.kernel_size());
				size_t feature_index = 0;
				for (size_t row = 0; row < kernel_moves; ++row)
				{
					for (size_t col = 0; col < kernel_moves; ++col)
					{
						//apply stride here
						const size_t start = ((row*_conv_params.stride()) * _output_width) + (col*_conv_params.stride());
						size_t kernel_index = 0;
						for (size_t i = 0; i < _conv_params.stride(); ++i)
						{
							const size_t local_start = start + (i * _output_width);
							const size_t local_end = local_start + _conv_params.kernel_width();
							kernel_index = i * _conv_params.kernel_width();
							for (size_t j = local_start; j < local_end; ++j)
							{
								for (size_t k = 0; k < input_depth; ++k)
								{
									_cache(current_kernel, j) += kernel_weight(k, kernel_index)  * (*input)(k, feature_index);
								}
								++kernel_index;

							}
						}
						++feature_index;
					}
				}
				_cache.row(current_kernel).array() += _biases(current_kernel);
			}
			_output = _activation->forward(_cache);
		}


		inline void correlate2d_t_bwd(MatrixRm* input, MatrixRm* dout) {
			const size_t input_depth = input->rows();
			const size_t delta_width = (size_t)sqrt(dout->cols());
			const size_t kernel_moves = (size_t)((delta_width - _conv_params.kernel_width()) / _conv_params.stride()) + 1;
			dout->array() *= _activation->backward(_cache).array();
			for (unsigned int current_kernel = 0; current_kernel < _num_kernels; ++current_kernel)
			{
				MatrixRm kernel_gradient = MatrixRm::Zero(_num_channels, _conv_params.kernel_size());
				MatrixRm kernel_weight = Eigen::Map<MatrixRm>(_weights.row(current_kernel).data(), _num_channels, _conv_params.kernel_size());
				size_t feature_index = 0;
				for (size_t row = 0; row < kernel_moves; ++row)
				{
					for (size_t col = 0; col < kernel_moves; ++col)
					{
						const size_t start = ((row*_conv_params.stride()) * delta_width) + (col*_conv_params.stride());
						size_t kernel_index = 0;
						for (size_t i = 0; i < _conv_params.stride(); ++i)
						{
							const size_t local_start = start + (i * delta_width);
							const size_t local_end = local_start + _conv_params.kernel_width();
							for (size_t j = local_start; j < local_end; ++j)
							{
								for (size_t k = 0; k < input_depth; ++k)
								{
									kernel_gradient(k, kernel_index) += (*dout)(current_kernel, j) * (*input)(k, feature_index);
									if (_previous_layer->get_layer_type() != INPUT_LAYER)
									{
										_dx(k, feature_index) += (*dout)(current_kernel, j) * kernel_weight(k, kernel_index);
									}
								}
								++kernel_index;
							}
						}
						++feature_index;
					}
				}
				_bias_gradients(current_kernel) += (*dout).row(current_kernel).sum();
				_gradients.row(current_kernel).array() += Eigen::Map<MatrixRm>(kernel_gradient.data(), 1, _weights.row(current_kernel).cols()).array();
			}
		}

		inline void correlate2d_t_fwd_gemm(MatrixRm* input)
		{
			const size_t row_len = (size_t)(input->cols() * input->rows()) / _num_kernels;
			MatrixRm im2col = _weights.transpose() * MatrixRmMap(input->transpose().data(), _weights.rows(), row_len);
			col2im(&_output, &im2col, &_cache, _conv_params);
			for (size_t i = 0; i < _num_kernels; i++)
			{
				_cache.row(i).array() += _biases(i, 0);
			}
			_output = _activation->forward(_cache);

		}

	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor for Conv2dLayer
			*/
			Conv2dTransposeLayer(std::shared_ptr<Layer>& previous_layer, const size_t out, const size_t kernel_size, size_t stride, std::shared_ptr<Activation> activation = nullptr, std::shared_ptr<Initializer> initializer = nullptr, std::string layer_name = "Conv2dLayerTransposed") : Conv2dLayer(previous_layer, out, kernel_size, stride, VALID, activation, initializer, layer_name)
		{
			_layer_type = CONV2D_T_LAYER;
			init();
		}


		/*
		* Performs forward pass of conv2d layer.
		* TODO: If possible, try to avoid looping over rows for adding biases. This also applies to the GEMM version of cc.
		*/
		void forward(bool training = false) override
		{
            (void) training;
			/*
			Ensure that cache and output is set to zero
			*/
			//_cache.setZero();
			//_output.setZero();
			correlate2d_t_fwd(_previous_layer->get_output());
		}

		/*
		* Performs backward pass of Conv2dlayer
		*/
		void backward(MatrixRm* dout) override
		{
			/*
			Ensure _dx is set to zero
			*/
			_dx.setZero();
			merge_deltas(dout);
			correlate2d_t_bwd(_previous_layer->get_output(), dout);
		}

		/*
		* Resets layer's gradients (bias and weights). Here it's overriden to update filters in case that winograd2d is used.
		*/
		void reset_gradients() override {
			_gradients.setZero();
			_bias_gradients.setZero();
		}

        static inline std::shared_ptr<Layer> create(std::shared_ptr<Layer> previous_layer, const size_t num_kernels, const size_t kernel_width, const size_t stride, ActivationType activation = RELU, InitializerType initializer = HE_NORMAL)
        {
            return std::move(std::make_shared<Conv2dTransposeLayer>(previous_layer, num_kernels, kernel_width, stride, Activation::get_activation_ptr(activation), Initializer::get_initializer_ptr(initializer)));
        }

	};

	/*
	*
	* Class defintion for 2d convolutional layer, but not actual convolution is performed. It's actually a cross-correlation.
	* The memory layout is the following:
	* Input:				R^2 here, the input is a matrix where every row is the channel of an image or a feature map. An NxN single channel image
	*						is here represented as 1xN*N matrix.
	* Weights, Gradients:	The weights and associated gradients are a bit weird but serve the purpose of performaing the cross-correlation as GEMM.
	Hence, it's a NxM Matrix where N is the number of filters and M a C*K*K. Where C is the number of input channels and K*K
	the kernel size, e.g. 3x3 = 9.
	*
	*/
	class Conv2dConcatLayer : public Conv2dLayer
	{

	private:
		size_t _offset;
		size_t _original_depth;
		size_t _crop_layer_width;
		size_t _crop_layer_rows;
		MatrixRm _crop;
		MatrixRm _concat;

		/*
		 * TODO: Avoid multiple copies here...
		 *
		 */
		inline void crop_and_concat()
		{
			crop(_fwd_skip_connection->get_output(), &_crop, _offset);
			_concat.block(0, 0, _crop.rows(), _crop.cols()) = _crop;
			_concat.block(_crop.rows(), 0, _previous_layer->get_output()->rows(), _crop.cols()) = *_previous_layer->get_output();
		}

		void init() override
		{
			_input_width = (size_t)sqrt(_previous_layer->get_output_size());
			_num_channels = _crop_layer->get_output_channels() + _previous_layer->get_output_channels();
			_crop_layer_width = (size_t)sqrt(_crop_layer->get_output_size());
			_crop_layer_rows = (size_t)_previous_layer->get_output_channels();
			_offset = _crop_layer_width - _input_width;
			_concat = MatrixRm::Zero(_num_channels, _input_width * _input_width);
			_crop = MatrixRm::Zero(_crop_layer->get_output()->rows(), _input_width * _input_width);
			_weights = MatrixRm::Zero(_num_kernels, _conv_params.kernel_size() * _num_channels);
			_gradients = MatrixRm::Zero(_num_kernels, _conv_params.kernel_size() * _num_channels);
			_original_depth = _previous_layer->get_output()->rows();
			_dx = MatrixRm::Zero(_original_depth, _previous_layer->get_output_size());
			_im2col_input = MatrixRm::Zero(_num_channels*_conv_params.kernel_size(), _output_width*_output_width);
			_output_size = (size_t)pow(_output_width, 2.);
			_output_channels = _num_kernels;
			_skip_dx = MatrixRm::Zero(_crop_layer->get_output_channels(), _crop_layer->get_output_size());
			initialize_parameters();
		}

		/*
		* Naive convolution backward operation. This is actually a cross-correlation.
		* TODO: Put some more love into it.
		*/
		inline void correlate2d_backwards_crop(MatrixRm* input, MatrixRm* dout) {
			const size_t input_depth = input->rows();
			for (unsigned int current_kernel = 0; current_kernel < _weights.rows(); current_kernel++)
			{
				MatrixRm kernel_gradient = MatrixRm::Zero(_num_channels, _conv_params.kernel_size());
				MatrixRm kernel_weight = Eigen::Map<MatrixRm>(_weights.row(current_kernel).data(), _num_channels, _conv_params.kernel_size());
				size_t feature_index = 0;
				for (size_t row = 0; row < _output_width; ++row)
				{
					for (size_t col = 0; col < _output_width; ++col)
					{
						(*dout)(current_kernel, feature_index) *= _activation->backward(_cache(current_kernel, feature_index));
						const size_t start = ((row*_conv_params.stride()) * _input_width) + (col*_conv_params.stride());
						size_t kernel_index = 0;
						for (size_t i = 0; i < _conv_params.kernel_width(); ++i)
						{
							const size_t local_start = start + (i * _input_width);
							const size_t local_end = local_start + _conv_params.kernel_width();
							kernel_index = i * _conv_params.kernel_width();
							for (size_t j = local_start; j < local_end; ++j)
							{
								for (size_t k = 0; k < input_depth; ++k)
								{
									kernel_gradient(k, kernel_index) += (*dout)(current_kernel, feature_index) * (*input)(k, j);

									if (_previous_layer->get_layer_type() != INPUT_LAYER && k >= _crop_layer_rows)
									{
										_dx(k - _crop_layer->get_output()->rows(), j) += (*dout)(current_kernel, feature_index) * kernel_weight(k, kernel_index);
									}
								}
								++kernel_index;
							}
						}
						++feature_index;
					}
				}
				_gradients.row(current_kernel).array() += Eigen::Map<MatrixRm>(kernel_gradient.data(), 1, _weights.row(current_kernel).cols()).array();
				_bias_gradients(current_kernel) += (*dout).row(current_kernel).sum();
			}
		}


		inline void correlate2d_c_bwd_gemm(MatrixRm* input, MatrixRm* dout)
		{
			dout->array() *= _activation->backward(_cache).array();
			MatrixRm wgrads;
			MatrixRm dout_rs;
			compute_wgrad_gemm_cpu(dout, &_im2col_input, &wgrads, &dout_rs, _weights.rows(), _weights.cols(), _num_kernels);
			_gradients.array() += wgrads.array();
			if (_previous_layer->get_layer_type() != INPUT_LAYER)
			{
				MatrixRm dx;
				compute_dx_gemm_cpu(&_dx, &dout_rs, input, &_weights, _conv_params);
				unsigned int row = 0;
				for (int i = _crop_layer->get_output()->rows(); i < _cache.rows(); ++i)
				{
					_dx.row(row) = dx.row(i);
				}
			}
			for (size_t i = 0; i < _num_kernels; ++i)
			{
				_bias_gradients(i, 0) = dout->row(i).sum();
			}
		}

	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor for Conv2dLayer
			*/
			Conv2dConcatLayer(std::shared_ptr<Layer> previous_layer, std::shared_ptr<Layer> crop_layer, const size_t out, const size_t kernel_size, size_t stride, std::shared_ptr<Activation> activation = nullptr, std::shared_ptr<Initializer> initializer = nullptr, std::string layer_name = "Conv2dConcatLayer") : Conv2dLayer(previous_layer, out, kernel_size, stride, VALID, activation, initializer, layer_name)
		{
			_layer_type = CONV2D_CC_LAYER;
			_crop_layer = crop_layer;
			//Add backwards connection to crop layer
			init();
		}


		/*
		* Performs forward pass of conv2d layer.
		* TODO: If possible, try to avoid looping over rows for adding biases. This also applies to the GEMM version of cc.
		*/
		void forward(bool training = false) override
		{
            (void) training;
			crop_and_concat();
			correlate2d_fwd_gemm(&_concat);
		}

		/*
		* Returns shared_ptr to crop layer
		*/
		std::shared_ptr<Layer> get_crop_layer()
		{
			return this->_crop_layer;
		}

		/*
		* Performs backward pass of Conv2dlayer
		*/
		void backward(MatrixRm* dout) override
		{
			_dx.setZero();
			merge_deltas(dout);
			correlate2d_bwd_gemm(&_concat, dout);
			//Layer must ensure that dimensions to skip layer match
            for (int i = 0; i < _crop.rows(); ++i) {
                MatrixRmMap(_skip_dx.row(i).data(), _crop_layer_width, _crop_layer_width).block(_offset / 2, _offset / 2, _input_width, _input_width) =
                        MatrixRmMap(_dx.row(i).data(), _input_width, _input_width);
            }
            //Strip cropped part from dx
			//TODO: Find better solution to drop first k channels from Matrix
            MatrixRm tmp = _dx.block(_crop.rows(), 0, _previous_layer->get_output_channels(), _previous_layer->get_output_size());
			_dx = tmp;
		}

        static inline std::shared_ptr<Layer> create(std::shared_ptr<Layer> previous_layer, std::shared_ptr<Layer> crop_layer, const size_t num_kernels, const size_t kernel_width, const size_t stride, ActivationType activation = RELU, InitializerType initializer = HE_NORMAL)
        {
            return std::move(std::make_shared<Conv2dConcatLayer>(previous_layer, crop_layer, num_kernels, kernel_width, stride, Activation::get_activation_ptr(activation), Initializer::get_initializer_ptr(initializer)));
        }
	};


	/*
	* Class definition for MaxPool2d
	*/
	class MaxPool2dLayer : public Layer
	{
	private:
		MatrixRm _pool_mask;
		size_t _window_size;
		size_t _window_moves;
		size_t _num_channels;
		size_t _input_width;
		size_t _stride;
		MatrixRm _tile;

		/*
		* Init method for 2d maxpooling layer
		*/
		void init() override
		{
			_num_channels = _previous_layer->get_output_channels();
			_output_channels = _num_channels;
			_dx = MatrixRm::Zero(_num_channels, _previous_layer->get_output_size());
			_input_width = (size_t)sqrt(_previous_layer->get_output_size());
			_window_moves = (size_t)floor((_input_width - _window_size) / _stride) + 1;
			_output_size = (size_t)pow(_window_moves, 2.);
			_output = MatrixRm::Zero(_num_channels, _output_size);
			_cache = MatrixRm::Zero(_num_channels, _output_size);
			_pool_mask = MatrixRm::Zero(_num_channels, _output_size);
			_tile = MatrixRm::Zero(_window_size, _window_size);
		}

		/*
		* Performs 2d maxpooling.
		*/
		inline void maxpool2d(const MatrixRm* input) {
			const size_t num_features = input->rows();
			const size_t window_moves = floor((_input_width - _window_size) / _stride) + 1;
			for (size_t current_feature = 0; current_feature < num_features; ++current_feature)
			{
				size_t feature_index = 0;
				for (size_t row = 0; row < window_moves; ++row)
				{
					const size_t current_row = row * _input_width;
					for (size_t col = 0; col < window_moves; ++col)
					{
						const size_t current_pos = (current_row*_stride) + (col*_stride);
						for (size_t kernel_row = 0; kernel_row < _window_size; ++kernel_row)
						{
							_tile.block(kernel_row, 0, 1, _window_size) = (*input).block(current_feature, current_pos + kernel_row*_input_width, 1, _window_size);
						}
						unsigned int i = 0, j = 0;
						const float max = _tile.maxCoeff(&i, &j);
						_pool_mask(current_feature, feature_index) = current_pos + i*_input_width + j;
						_output(current_feature, feature_index) = max;
						++feature_index;
					}
				}
			}
		}


	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor of maxpool 2d layer
			*/
			MaxPool2dLayer(std::shared_ptr<Layer>& previous_layer, const size_t window_size, const size_t stride = 2, std::string layer_name = "MaxPool2dLayer") : Layer(previous_layer, layer_name)
		{
			_layer_type = MAXPOOL2D_LAYER;
			_window_size = window_size;
			_stride = stride;
			init();
		}
		/*
		* Getter for stride.
		*/
		size_t get_stride()
		{
			return _stride;
		}

		/*
		* Getter for window size.
		*/
		size_t get_window_size()
		{
			return _window_size;
		}

		/*
		* Forward pass for MaxPool2d layer
		*/
		void forward(bool training = true) override
		{
			/*
			Probably find a better way to reset
			*/
            (void) training;
			_cache.setZero();
			_output.setZero();
			MatrixRm* input_features = get_previous_layer()->get_output();
			maxpool2d(input_features);
		}
		/*
		* Backwards pass of MaxPool2d layer
		*/
		void backward(MatrixRm* deltas) override
		{
			_dx.setZero();
			size_t delta_channels = (*deltas).rows();
			size_t delta_size = (*deltas).cols();
			for (size_t current_channel = 0; current_channel < delta_channels; ++current_channel)
			{
				for (size_t current_idx = 0; current_idx < delta_size; ++current_idx)
				{
					_dx(current_channel, _pool_mask(current_channel, current_idx)) = (*deltas)(current_channel, current_idx);
				}
			}
			_pool_mask.setZero();
		}

        static inline std::shared_ptr<Layer> create(std::shared_ptr<Layer> previous_layer, size_t window_size, size_t stride)
        {
            return std::move(std::make_shared<MaxPool2dLayer>(previous_layer, window_size, stride));
        }
	};


	/*
	* Class definition for DropoutLayer
	*/
	class DropoutLayer : public Layer
	{

	private:
		MatrixRm _dropout_mask;
		MatrixRm* _output_ptr;
		bool _is_training;
		bool _use_seed;
		int _seed;
		float _dropout_prob;
		std::mt19937 _rng;
		std::uniform_real_distribution<float> _dist;

		/*
		* Init method for dropout layer
		*/
		void init()
		{
			_output_channels = _previous_layer->get_output_channels();
			_output_size = _previous_layer->get_output_size();
			_is_training = false;
			if (this->_use_seed)
			{
				this->_rng.seed(this->_seed);
			}
			else
			{
				this->_rng.seed(std::random_device()());
			}
			this->_dropout_mask = MatrixRm::Zero(this->_previous_layer->get_output()->rows(), this->_previous_layer->get_output()->cols());
			_dist = std::uniform_real_distribution<float>(0.f, 1.f);
			_output = MatrixRm::Zero(_previous_layer->get_output()->rows(), _previous_layer->get_output()->cols());
		}

		inline void dropout_fwd(MatrixRm* input)
		{
			_dropout_mask = (_dropout_mask.unaryExpr([&](float x)
			{
				(void)x;
				return _dist(_rng);
			}
			).array() > _dropout_prob).cast<float>();
			this->_output = input->array() * this->_dropout_mask.array();
		}

		inline void dropout_bwd(MatrixRm* dout)
		{
			this->_dx = dout->array() * this->_dropout_mask.array();
		}


	public:

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			/*
			* Constructor of DropoutLayer
			*/
			DropoutLayer(std::shared_ptr<Layer>& previous_layer, const float dropout_prob, std::string layer_name = "DropoutLayer") : Layer(previous_layer, layer_name)
		{
			_layer_type = DROP_LAYER;
			this->_dropout_prob = dropout_prob;
			init();
		}
		/*
		* Getter for droput prob.
		*/
		float get_dropout_prob()
		{
			return _dropout_prob;
		}

		MatrixRm* get_output() override
		{
			if (_is_training)
			{
				return &_output;
			}
			else
			{
				return _output_ptr;
			}
		}

		/*
		* Forward pass for DropoutLayer
		*/
		void forward(bool is_training = false) override
		{
			_is_training = is_training;
			if (_is_training)
			{
				_output_ptr = this->get_previous_layer()->get_output();
			}
			else
			{
				dropout_fwd(this->get_previous_layer()->get_output());
			}
		}
		/*
		* Backwards pass of DropoutLayer
		*/
		void backward(MatrixRm* dout) override
		{
			_dx.setZero();
			dropout_bwd(dout);
		}

		static inline std::shared_ptr<Layer> create(std::shared_ptr<Layer> previous_layer, const float dropout_prob)
        {
		    return std::move(std::make_shared<DropoutLayer>(previous_layer, dropout_prob));
        }
	};
}

#endif