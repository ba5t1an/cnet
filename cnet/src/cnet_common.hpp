/**
Common functions and typedefs.

@file cnet_common.hpp
@author Bastian 
*/


#ifndef CNET_INTERNAL_HPP
#define CNET_INTERNAL_HPP

#include <map>
#include <fstream> 

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <experimental/filesystem>
#include <mutex>
#include <condition_variable>
#include "cnet_cv2eigen.hpp"


#define INITIAL_QUEUE_SIZE 64
#define THREAD_RETRY_DELAY 2



namespace Cnet
{

	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRm;
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixCm;
	typedef std::vector<MatrixRm, Eigen::aligned_allocator<MatrixRm > > AlignedStdVector;
	typedef std::vector<MatrixRm, Eigen::aligned_allocator<MatrixRm > > AlignedStdVector;
	//typedef std::vector<std::pair<MatrixRm, MatrixRm>, std::pair< Eigen::aligned_allocator< MatrixRm >, Eigen::aligned_allocator< MatrixRm > >  > AlignedStdPairVector;
	typedef std::map<int, Eigen::Vector4f, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Vector4f> > > AlignedMap;
	typedef Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > MatrixRmMap;
	typedef Eigen::Map< Eigen::VectorXf> VectorMap;

	inline void disable_multithreading()
	{
		Eigen::setNbThreads(1);
	}

	/*
	Checks if file exists...
	*/
	inline bool file_exists(const std::string& file_name)
	{
		std::ifstream infile(file_name);
		return infile.good();
	}

	/*
	Get filename from string
	*/
	inline std::string strip_filename(const std::string& path)
	{
		std::string stripped = path.substr(path.find_last_of("/\\") + 1);
		size_t dot_i = path.find_last_of('.');
		return stripped.substr(0, dot_i);
	}

	/*
	Struct to hold convolution parameters
	*/
	class ConvolutionParams
	{
	private:
		size_t _kernel_size;
		size_t _kernel_width;
		size_t _stride;
		size_t _padding;
	public:

		ConvolutionParams(size_t kernel_width, size_t stride, size_t padding)
		{
			_kernel_width = kernel_width;
			_kernel_size = (size_t)pow(kernel_width, 2.0);
			_stride = stride;
			_padding = padding;
		
		}

		inline size_t kernel_size()
		{
			return _kernel_size;
		}

		inline size_t padding()
		{
			return _padding;
		}

		inline size_t kernel_width()
		{
			return _kernel_width;
		}

		inline size_t stride()
		{
			return _stride;
		}

		void stride(size_t stride)
		{
			_stride = stride;
		}

		void padding(size_t padding)
		{
			_padding = padding;
		}

		void kernel_width(size_t kernel_width)
		{
			_kernel_width = kernel_width;
			_kernel_size = (size_t)pow(kernel_width, 2.0);
		}

	};

	/*
	Struct to hold Datasample
	*/
	class Entry
	{
	private:
		MatrixRm _data;
		bool _is_file;
		std::string _file_name;
		std::string _file_path;
	public:

		Entry(MatrixRm data, const std::string& fpath)
		{
			_data = data;
			_file_name = strip_filename(fpath);
			_file_path = fpath;
			_is_file = true;
		}

        explicit Entry(MatrixRm data)
		{
			_data = data;
			_is_file = false;
		}

		Entry()
		{
			_is_file = false;
		}

		MatrixRm* data()
		{
			return &_data;
		}

		std::string file_name()
		{
			return _file_name;
		}

		bool is_file()
		{
			return _is_file;
		}

		std::string file_path()
		{
			return _file_path;
		}

	};

	/*
	Template for thread-safe queue class
	*/
	template <class T> class DataQueue
	{
	public:

        explicit DataQueue(size_t max_size)
		{
			_max_size = max_size;
			_current_size = 0;
		}

		DataQueue()
		{
			_max_size = INITIAL_QUEUE_SIZE;
			_current_size = 0;
		}

		~DataQueue() = default;


		void push(T t)
		{
			std::unique_lock<std::mutex> lock(_m);
			_c.wait(lock, [this]() {return _queue.size() < _max_size; });
			_queue.push_back(t);
			_c.notify_all();
		}

		void push_streight(T t)
		{
			//std::unique_lock<std::mutex> lock(_m);
			//c.wait(lock, [this]() {return _queue.size() < _max_size; });
			_queue.push_back(t);
			//_c.notify_all();
		}

		T pop()
		{
			std::unique_lock<std::mutex> lock(_m);
			while (_queue.empty())
			{
				_c.wait(lock);
			}
			_c.notify_all();
			T val = _queue.front();
			_queue.pop_front();
			return val;
		}

		bool try_pop(T& item, std::chrono::milliseconds timeout)
		{
			std::unique_lock<std::mutex> lock(_m);
			if (!_c.wait_for(lock, timeout, [this] { return !_queue.empty(); }))
			{
				return false;
			}
			_c.notify_all();
			item = _queue.front();
			_queue.pop_front();
			return true;
		}

		T at(size_t idx)
		{
			std::unique_lock<std::mutex> lock(_m);
			while (_queue.empty())
			{
				_c.wait(lock);
			}
			_c.notify_all();
			T val = _queue[idx];
			return val;
		}

		size_t get_size()
		{
			return _queue.size();
		}

		size_t get_max_size()
		{
			return _max_size;
		}

		void set_max_size(size_t queue_size)
		{
			_max_size = queue_size;
		}

	private:
		std::deque<T> _queue;
		mutable std::mutex _m;
		std::condition_variable _c;
		size_t _max_size;
		size_t _current_size;
	};
	

	inline int load_image(MatrixRm* image, const std::string& path)
	{
		if (file_exists(path))
		{
			MatrixRm input;
			eigen2cv(input) = cv::imread(path, cv::IMREAD_GRAYSCALE);
			//scale down
			*image = MatrixRmMap(input.data(), 1, input.cols()*input.rows());
			//*image /= 255.f;
			//std::cout << "image:\n" << *image << std::endl;
			return 1;
		}
		return 0;
	}

	inline int load_image(MatrixRm& image, const std::string& path)
	{
		return load_image(&image, path);
	}

	void save_image(MatrixRm* image, std::string path)
	{
		auto width = (size_t) sqrt(image->cols());
		MatrixRm out = MatrixRmMap(image->data(), width, width);
		out *= 255;
		cv::Mat mat_out = eigen2cv(out);
		cv::imwrite(path, mat_out);
	}

	void save_image(MatrixRm& image, const std::string& path)
	{
		save_image(&image, path);
	}

	/*
	Method to extract image tiles.
	*/
	inline void extract_image_tiles(MatrixRm* image, AlignedStdVector tile_vec, const size_t tile_size, const size_t stride)
	{
		MatrixRm tile = MatrixRm::Zero(tile_size, tile_size);
		const auto input_width = (size_t)sqrt(image->cols());
		const auto num_tiles = (size_t)((image->cols() - tile_size) / 2) + 1;
		for (size_t row = 0; row < num_tiles; ++row)
		{
			const size_t current_row = row * input_width;
			for (size_t col = 0; col < num_tiles; ++col)
			{
				const size_t current_pos = (current_row*stride) + (col*stride);
				for (size_t kernel_row = 0; kernel_row < tile_size; ++kernel_row)
				{
					tile.block(kernel_row, 0, 1, tile_size) = (*image).block(0, current_pos + kernel_row*input_width, 1, tile_size);
				}
				tile_vec.push_back(tile);
			}
		}
	}

	/*
	Method to pad border with zeros.
	*/
	inline void pad_border(MatrixRm* in, MatrixRm* out, size_t offset)
	{
		if (offset == 0)
		{
			*out = *in;
		}
		else
		{
			const auto input_width = (size_t)sqrt(in->cols());
			*out = MatrixRm::Zero(in->rows(), (unsigned int)pow(2 * offset + input_width, 2.));
			const size_t steps = 2 * offset + input_width;
			size_t start = 0;
			for (size_t i = 0; i < input_width; ++i)
			{
				out->block(0, start, in->rows(), input_width) = in->block(0, i*input_width, in->rows(), input_width);
				start += steps;
			}
		}
	}


	/*
	Alternative method to add zeros to the border.
	*/
	inline void pad(MatrixRm* in, MatrixRm* out, size_t offset)
	{
		const size_t padded_width = (size_t)sqrt(in->cols()) + offset;
		*out = MatrixRm::Zero(in->rows(), (unsigned int)pow(padded_width, 2.));
		const auto org_width = (size_t)sqrt(in->cols());
		for (unsigned int channel = 0; channel < out->rows(); ++channel)
		{
			MatrixRmMap(out->row(channel).data(), padded_width, padded_width).block(offset /2, offset/2, org_width, org_width) 
				= MatrixRmMap(in->row(channel).data(), org_width, org_width);
		}
	}

	/*
	Crops center of the image by given offset.
	*/
	inline void crop(MatrixRm* in, MatrixRm* out, size_t offset)
	{
		const size_t cropped_width = (size_t)sqrt(in->cols()) - offset;
		*out = MatrixRm(in->rows(), (unsigned int)pow(cropped_width, 2.));
		const auto org_width = (size_t)sqrt(in->cols());
		for (unsigned int channel = 0; channel < out->rows(); ++channel)
		{
			MatrixRmMap(out->row(channel).data(), cropped_width, cropped_width) = MatrixRmMap(in->row(channel).data(), org_width, org_width).block(offset / 2, offset / 2, cropped_width, cropped_width);
		}
	}

	/*
	Inserts zeros between values of a matrix by stride - 1.
	*/
	inline void pad_inner(MatrixRm* input, MatrixRm* out, const size_t stride)
	{
		const auto input_width = (size_t)sqrt(input->cols());
		const size_t inner_pad = stride - 1;
		const size_t out_width = input_width + (input_width*inner_pad - 1);
		*out = MatrixRm::Zero(input->rows(), (unsigned int)pow(out_width, 2.));
		size_t current_pos = 0;
		for (int i = 0; i < input->cols(); ++i)
		{
			out->block(0, current_pos, input->rows(), 1) = input->block(0, i, input->rows(), 1);
			if (++current_pos % out_width != 0)
			{
				current_pos += inner_pad;
			}
			else
			{
				current_pos += out_width;
			}
		}
	}

	/*
	Reverse's pad_inner method using stride - 1.
	*/
	inline void unpad_inner(MatrixRm* input, MatrixRm* out, const size_t org_width, const size_t stride)
	{
		const auto input_width = (size_t)sqrt(input->cols());
		const auto inner_pad = stride - 1;
		*out = MatrixRm::Zero(input->rows(), (unsigned int)pow(org_width, 2.));
		size_t current_pos = 0;
		for (int i = 0; i < out->cols(); ++i)
		{
			out->block(0, i, input->rows(), 1) = input->block(0, current_pos, input->rows(), 1);
			if (++current_pos % input_width != 0)
			{
				current_pos += inner_pad;
			}
			else
			{
				current_pos += input_width;
			}
		}
	}

	/*
	Performs im2col operation. As before, see Caffe's source code for further details. This can potentially implemented in a faster way....
	*/
	inline void im2col(MatrixRm* input, MatrixRm* out, const size_t kernel_width, const size_t kernel_size, const size_t stride)
	{
		const auto input_width = (size_t)sqrt(input->cols());
		const auto kernel_moves = (size_t)((input_width - kernel_width) / stride) + 1;
		*out = MatrixRm::Zero(input->rows()* kernel_size, (unsigned int)pow(kernel_moves, 2));
		size_t column_idx = 0;
		MatrixRm tile = MatrixRm::Zero(input->rows(), kernel_size);
		for (size_t row = 0; row < kernel_moves; ++row)
		{
			const size_t current_row = row * input_width;
			for (size_t col = 0; col < kernel_moves; ++col)
			{
				const size_t current_pos = (current_row*stride) + (col*stride);
				for (size_t kernel_row = 0; kernel_row < kernel_width; ++kernel_row)
				{
					tile.block(0, kernel_row*kernel_width, input->rows(), kernel_width) =
						input->block(0, current_pos + kernel_row*input_width, input->rows(), kernel_width);
				}
				out->col(column_idx++) = Eigen::Map<Eigen::VectorXf>(tile.data(), out->rows());
			}
		}
	}

	/*
	Performs im2col operation. See Caffe's source code for further details.
	*/
	inline void im2col(MatrixRm* input, MatrixRm* out, ConvolutionParams& conv_params)
	{
		im2col(input, out, conv_params.kernel_width(), conv_params.kernel_size(), conv_params.stride());
	}

	/*
	Reverse operation for im2col.
	*/
	inline void col2im(MatrixRm* input, const MatrixRm* im2col, MatrixRm* out, ConvolutionParams& conv_params)
	{
		const auto input_width = (size_t)sqrt(input->cols());
		const auto kernel_moves = (size_t)((input_width - conv_params.kernel_width()) / conv_params.stride()) + 1;
		*out = MatrixRm::Zero(input->rows(), input->cols());
		size_t col_cnt = 0;
		for (size_t row = 0; row < kernel_moves; ++row)
		{
			for (size_t col = 0; col < kernel_moves; ++col)
			{
				for (int ch = 0; ch < input->rows(); ++ch)
				{
					MatrixRm chunk = im2col->block(ch * conv_params.kernel_size(), col_cnt, conv_params.kernel_size(), 1);
					MatrixRmMap(out->row(ch).data(), input_width, input_width).block(row, col, conv_params.kernel_width(), conv_params.kernel_width()).array()
						+= MatrixRmMap(chunk.data(), conv_params.kernel_width(), conv_params.kernel_width()).array();
				}
				++col_cnt;
			}
		}
	}

	/*
	Alternative col2im method... still reversing the col2im.
	*/
	inline void col2im(MatrixRm* im2col, MatrixRm* out, const size_t output_width, ConvolutionParams& conv_params)
	{
		const size_t output_depth = (size_t)im2col->rows() / conv_params.kernel_size();
		*out = MatrixRm::Zero(output_depth, output_width * output_width);
		size_t row = 0;
		size_t col = 0;
		for (int i = 0; i < im2col->cols(); i++)
		{
			for (size_t j = 0; j < output_depth; j++)
			{
				MatrixRm chunk = im2col->block(j*conv_params.kernel_size(), 0, conv_params.kernel_size(), 1);
				MatrixRmMap(out->row(j).data(), output_width, output_width).block(col, row, conv_params.kernel_width(), conv_params.kernel_width()).array() +=
					MatrixRmMap(chunk.data(), conv_params.kernel_width(), conv_params.kernel_width()).array();

			}
			col += conv_params.stride();
			if ((col + conv_params.kernel_width()) > output_width)
			{
				col = 0;
				row += conv_params.stride();
			}
		}
	}

	/*
	Computes weight gradient as matrix multiplication.
	*/
	inline void compute_wgrad_gemm_cpu(MatrixRm* dout, MatrixRm* im2col_input, MatrixRm* wgrad, MatrixRm* dout_rs, const size_t w_rows, const size_t w_cols, const size_t num_filters)
	{
		const size_t row_len = (size_t)(dout->cols() * dout->rows()) / num_filters;
		*dout_rs = MatrixRmMap(dout->transpose().data(), w_rows, row_len);
		MatrixRm wgrad_tmp = *dout_rs * im2col_input->transpose();
		*wgrad = MatrixRmMap(wgrad_tmp.data(), w_rows, w_cols).array();
	}

	/*
	Computes input delta as matrix multiplication.
	*/
	inline void compute_dx_gemm_cpu(MatrixRm* dx, MatrixRm* dout_rs, MatrixRm* input, MatrixRm* w, ConvolutionParams& conv_params)
	{
		MatrixRm dx_col = w->transpose() * *dout_rs;
		col2im(input, &dx_col, dx, conv_params);
	}

	/*
	Computes backwards pass of a convolution layer as matrix multiplication...probably the wrong place for this.
	*/
	inline void backward_gemm_cpu(MatrixRm* dout, MatrixRm* dx, MatrixRm* input, MatrixRm* im2col_input, MatrixRm* w, MatrixRm* wgrad, ConvolutionParams& conv_params)
	{

		const size_t row_len = (size_t)(dout->cols() * dout->rows()) / wgrad->rows();
		MatrixRm dout_rs = MatrixRmMap(dout->transpose().data(), wgrad->rows(), row_len);
		MatrixRm raw_grads = dout_rs * im2col_input->transpose();
		*wgrad = MatrixRmMap(raw_grads.data(), wgrad->rows(), wgrad->cols());
		MatrixRm dx_col = w->transpose() * dout_rs;
		col2im(input, &dx_col, dx, conv_params);
	}

	inline void crop_center(MatrixRm* in, MatrixRm* out, size_t offset)
	{
		const auto input_width = (size_t)sqrt(in->cols());
		const size_t delta = input_width - (2 * offset);
		const size_t border = 2 * offset;
		*out = MatrixRm(in->rows(), (size_t)pow(delta, 2.));
		size_t start = (offset * input_width) + offset;
		for (size_t i = 0; i < delta; ++i)
		{
			out->block(0, i*delta, in->rows(), delta) = in->block(0, start, in->rows(), delta);
			start += delta + border;
		}
	}

	inline void pad_border(MatrixRm& in, MatrixRm& out, size_t offset)
	{
		const auto input_width = (size_t)sqrt(in.cols());
		const size_t steps = 2 * offset + input_width;
		size_t start = 0;
		for (size_t i = 0; i < input_width; ++i)
		{
			out.block(0, start, in.rows(), input_width) = in.block(0, i*input_width, in.rows(), input_width);
			start += steps;
		}
	}

	inline MatrixRm encode_one_hot_axis1(const size_t class_idx, const size_t num_classes)
	{
		MatrixRm one_hot_encoded_mat = MatrixRm::Zero(1, num_classes);
		for (size_t i = 0; i < num_classes; ++i)
		{
			if (class_idx == i)
			{
				one_hot_encoded_mat(0, i) = 1.f;
			}
		}
		return one_hot_encoded_mat;
	}


	/*
	* Function to encode labels to one-hot over x-axis, e.g. 2 -> [0, 1]
	* @param clas_idx
	* @param num_classes
	* @return the one-hot encoded vector
	*/
	inline MatrixRm encode_one_hot(const size_t class_idx, const size_t num_classes)
	{
		MatrixRm one_hot_encoded_mat = MatrixRm::Zero(1, num_classes);
		for (size_t i = 0; i < num_classes; ++i)
		{
			if (class_idx == i)
			{
				one_hot_encoded_mat(0, i) = 1.f;
			}
		}
		return one_hot_encoded_mat;
	}

	inline void encode_one_hot(MatrixRm& label, const size_t class_idx, const size_t num_classes)
	{
		label = MatrixRm::Zero(1, num_classes);
		for (size_t i = 0; i < num_classes; ++i)
		{
			if (class_idx == i)
			{
				label(0, i) = 1.f;
			}
		}
	}

	/*
	* Function to encode labels to one-hot over y-axis, e.g. 2 -> [0, 1]
	* @param clas_idx
	* @param num_classes
	* @return the one-hot encoded matrix
	*/
	/*
	MatrixRm encode_one_hot(const MatrixRm& label, const size_t num_classes)
	{
		MatrixRm one_hot_encoded_mat = MatrixRm::Zero(num_classes, label.cols());
		for (int i = 0; i < label.cols(); ++i)
		{
			for (size_t j = 0; j < num_classes; ++j)
			{
				if ((size_t)label(0, i) == j)
				{
					one_hot_encoded_mat(j, i) = 1.f;
				}
			}
		}
		return one_hot_encoded_mat;
	}
	*/

	inline void encode_label(AlignedStdVector& data_container, const size_t num_classes, const float label)
	{
		if (num_classes > 1)
		{
			MatrixRm label_mat = encode_one_hot((size_t)label, num_classes);
			data_container.push_back(label_mat);
		}
		else
		{
			MatrixRm label_mat(1, 1);
			label_mat(0, 0) = label;
			data_container.push_back(label_mat);
		}
	}

	/*
	* Argmax of a given input. Will determine the axis to perform action by the shape of outputs.
	* NOTE: Currently only 1d data is processed correctly.
	*/
	inline unsigned int argmax(MatrixRm* outputs)
	{
		unsigned int i = 0, j = 0;
		outputs->maxCoeff(&i, &j);
		return j;
	}

	inline unsigned int argmax(MatrixRm& outputs)
	{
		return argmax(&outputs);
	}

	inline void depthwise_argmax(MatrixRm& outputs, MatrixRm& argmaxed)
	{
		argmaxed = MatrixRm::Zero(1, outputs.cols());
		for (int col = 0; col < outputs.cols(); ++col)
		{
			unsigned int i = 0, j = 0;
			outputs.block(0, col, outputs.rows(), 1).maxCoeff(&i, &j);
			argmaxed(0, col) = (float)i;
		}
	}

	/*
	* Argmax of a given input. Will determine the axis to perform action by the shape of outputs.
	* NOTE: Currently only 1d data is processed correctly.
	*/
	inline void argmax(MatrixRm& outputs, MatrixRm& result)
	{
		for (unsigned int i = 0; i < outputs.cols(); ++i)
		{
			unsigned int x = 0, y = 0;
			outputs.block(0, i, outputs.rows(), 1).maxCoeff(&x, &y);
			result(0, i) = (float)y;
		}
	}

	inline MatrixRm to_matrix(std::vector<std::vector<float> > data)
	{
		MatrixRm sample = MatrixRm::Zero(data.size(), data[0].size());
		for (size_t i = 0; i < data.size(); i++)
		{
			for (size_t j = 0; j < data[i].size(); j++)
			{
				sample(i, j) = data[i][j];
			}
		}
		return sample;
	}

	/*
	MatrixRm to_matrix(std::vector<std::vector<float> > data)
	{
		MatrixRm sample = MatrixRm::Zero(data.size(), data[0].size());
		for (size_t i = 0; i < data.size(); i++)
		{
			for (size_t j = 0; j < data[i].size(); j++)
			{
				sample(i, j) = data[i][j];
			}
		}
		return sample;
	}
	*/
}
#endif
