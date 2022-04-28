/**
Implementation of winograds minimal filtering algorithm for convolution layers.

@file cnet_winograd2d.hpp
@author Bastian Schoettle EN RC PREC
*/

#ifndef CNET_WINOGRAD2D_HPP
#define CNET_WINOGRAD2D_HPP

#include "cnet_common.hpp"

/*
The classes here are experimental...even if probably working, consider not using them. 
GEMM version of conv2d is still faster....
*/
namespace Cnet 
{

	namespace Internal
	{

		static Eigen::Matrix<float, 4, 4, Eigen::RowMajor> trans_i4x4 = (Eigen::Matrix<float, 4, 4, Eigen::RowMajor> () << 1.f, 0.f, -1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 1.f, 0.f, -1.f).finished();
		static Eigen::Matrix<float, 4, 3, Eigen::RowMajor> trans_f3x3 = (Eigen::Matrix<float, 4, 3, Eigen::RowMajor>() << 1.f, 0.f, 0.f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.f, 0.f, 1.f).finished();
		static Eigen::Matrix<float, 2, 4, Eigen::RowMajor> trans_r2x2 = (Eigen::Matrix<float, 2, 4, Eigen::RowMajor>() << 1.f, 1.f, 1.f, 0.f, 0.f, 1.f, -1.f, -1.f).finished();
		
		static Eigen::Matrix<float, 4, 4> trans_i4x4_t = trans_i4x4.transpose();
		static Eigen::Matrix<float, 3, 4> trans_f3x3_t = trans_f3x3.transpose();
		static Eigen::Matrix<float, 4, 2> trans_r2x2_t = trans_r2x2.transpose();

		class Winograd2d
		{

		public:
			virtual void update(MatrixRm* filters) = 0;
			virtual void transform(MatrixRm* input, MatrixRm* output) = 0;

		protected:
			MatrixRm _v_matrix;
			MatrixRm _u_matrix;
			MatrixRm _uv_result;
			size_t _num_filters;
			size_t _num_channels;
			size_t _num_tiles;
			size_t _t_num_tiles;
			size_t _input_width;
		};

		class Winograd2d_4x4_3x3 : public Winograd2d
		{

		public:

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

			Winograd2d_4x4_3x3(MatrixRm* filters, size_t input_width) 
			{
				_input_width = input_width;
				_num_channels = (size_t) filters->cols() / 9;
				_num_filters = filters->rows();
				_v_matrix = MatrixRm::Zero(_num_channels, _num_filters * 16);
				_num_tiles = ((input_width - 4) / 2) + 1;
				_t_num_tiles = (size_t)pow(_num_tiles, 2.f);
				_u_matrix = MatrixRm::Zero(_t_num_tiles * 16, _num_channels);
				_uv_result = MatrixRm::Zero(_u_matrix.rows(), _v_matrix.cols());
				update(filters);

			}

			void update(MatrixRm* filters) override
			{
				for (size_t i = 0; i < _num_filters; ++i)
				{
					for (size_t j = 0; j < _num_channels; ++j)
					{
						MatrixRm tfilter = (trans_f3x3 * MatrixRmMap(filters->block(i, j * 9, 1, 9).data(), 3, 3)) * trans_f3x3_t;
						(_v_matrix)(j, 0 *_num_filters + i) = tfilter(0, 0);
						(_v_matrix)(j, 1 *_num_filters + i) = tfilter(0, 1);
						(_v_matrix)(j, 2 *_num_filters + i) = tfilter(0, 2);
						(_v_matrix)(j, 3 *_num_filters + i) = tfilter(0, 3);
						(_v_matrix)(j, 4 *_num_filters + i) = tfilter(1, 0);
						(_v_matrix)(j, 5 *_num_filters + i) = tfilter(1, 1);
						(_v_matrix)(j, 6 *_num_filters + i) = tfilter(1, 2);
						(_v_matrix)(j, 7 *_num_filters + i) = tfilter(1, 3);
						(_v_matrix)(j, 8 *_num_filters + i) = tfilter(2, 0);
						(_v_matrix)(j, 9 *_num_filters + i) = tfilter(2, 1);
						(_v_matrix)(j, 10 *_num_filters + i) = tfilter(2, 2);
						(_v_matrix)(j, 11 *_num_filters + i) = tfilter(2, 3);
						(_v_matrix)(j, 12 *_num_filters + i) = tfilter(3, 0);
						(_v_matrix)(j, 13 *_num_filters + i) = tfilter(3, 1);
						(_v_matrix)(j, 14 *_num_filters + i) = tfilter(3, 2);
						(_v_matrix)(j, 15 *_num_filters + i) = tfilter(3, 3);
					}
				}
			}

			void transform(MatrixRm* input, MatrixRm* output) override
			{

				transform_and_scatter_4x4s2_tiles(input);
				for (size_t i = 0; i < 16; ++i)
				{
					_uv_result.block(i*_t_num_tiles, 0, _t_num_tiles, _num_filters) = 
						_u_matrix.block(i*_num_tiles, 0, _t_num_tiles, _u_matrix.cols()) * 
						_v_matrix.block(0, i*_num_filters, _v_matrix.rows(), _num_filters);
				}
				reverse_transform(output);
			}

		private:

			inline void transform_and_scatter_4x4s2_tiles(MatrixRm* img)
			{
				for (size_t i = 0; i < _num_channels; ++i)
				{
				size_t tiles_cnt = 0;
				MatrixRm map = MatrixRmMap(img->row(i).data(), _input_width, _input_width);
				for (size_t j = 0; j < _num_tiles; ++j)
				{
					for (size_t k = 0; k < _num_tiles; ++k)
					{
							MatrixRm tin = (trans_i4x4* map.block(j * 2, k * 2, 4, 4)) * trans_i4x4_t;
							(_u_matrix)(0 * _t_num_tiles  + tiles_cnt, i) = tin(0, 0);
							(_u_matrix)(1 * _t_num_tiles  + tiles_cnt, i) = tin(0, 1);
							(_u_matrix)(2 * _t_num_tiles  + tiles_cnt, i) = tin(0, 2);
							(_u_matrix)(3 * _t_num_tiles  + tiles_cnt, i) = tin(0, 3);
							(_u_matrix)(4 * _t_num_tiles  + tiles_cnt, i) = tin(1, 0);
							(_u_matrix)(5 * _t_num_tiles  + tiles_cnt, i) = tin(1, 1);
							(_u_matrix)(6 * _t_num_tiles  + tiles_cnt, i) = tin(1, 2);
							(_u_matrix)(7 * _t_num_tiles  + tiles_cnt, i) = tin(1, 3);
							(_u_matrix)(8 * _t_num_tiles  + tiles_cnt, i) = tin(2, 0);
							(_u_matrix)(9 * _t_num_tiles  + tiles_cnt, i) = tin(2, 1);
							(_u_matrix)(10 * _t_num_tiles  + tiles_cnt, i) = tin(2, 2);
							(_u_matrix)(11 * _t_num_tiles  + tiles_cnt, i) = tin(2, 3);
							(_u_matrix)(12 * _t_num_tiles  + tiles_cnt, i) = tin(3, 0);
							(_u_matrix)(13 * _t_num_tiles  + tiles_cnt, i) = tin(3, 1);
							(_u_matrix)(14 * _t_num_tiles  + tiles_cnt, i) = tin(3, 2);
							(_u_matrix)(15 * _t_num_tiles  + tiles_cnt, i) = tin(3, 3);
						}
						++tiles_cnt;
					}
				}
			}

			void reverse_transform(MatrixRm* out)
			{
				const size_t out_width = sqrt(out->cols());
				MatrixRm res = MatrixRm::Zero(2, 2);
				MatrixRm tile = MatrixRm::Zero(4, 4);
				const size_t stride = 2;
				for (size_t i = 0; i < _num_filters; ++i)
				{
					size_t pos = 0;
					for (size_t j = 0; j < _t_num_tiles; ++j)
					{
						tile(0, 0) = (_uv_result)(0 * _t_num_tiles, 0);
						tile(0, 1) = (_uv_result)(1 * _t_num_tiles + j, i);
						tile(0, 2) = (_uv_result)(2 * _t_num_tiles + j, i);
						tile(0, 3) = (_uv_result)(3 * _t_num_tiles + j, i);
						tile(1, 0) = (_uv_result)(4 * _t_num_tiles + j, i);
						tile(1, 1) = (_uv_result)(5 * _t_num_tiles + j, i);
						tile(1, 2) = (_uv_result)(6 * _t_num_tiles + j, i);
						tile(1, 3) = (_uv_result)(7 * _t_num_tiles + j, i);
						tile(2, 0) = (_uv_result)(8 * _t_num_tiles + j, i);
						tile(2, 1) = (_uv_result)(9 * _t_num_tiles + j, i);
						tile(2, 2) = (_uv_result)(10 * _t_num_tiles + j, i);
						tile(2, 3) = (_uv_result)(11 * _t_num_tiles + j, i);
						tile(3, 0) = (_uv_result)(12 * _t_num_tiles + j, i);
						tile(3, 1) = (_uv_result)(13 * _t_num_tiles + j, i);
						tile(3, 2) = (_uv_result)(14 * _t_num_tiles + j, i);
						tile(3, 3) = (_uv_result)(15 * _t_num_tiles + j, i);
						res.noalias() = (trans_r2x2 * tile) * trans_r2x2_t;
						out->block(i, pos, 1, 2) = res.row(0);
						out->block(i, pos + out_width, 1, 2) = res.row(1);
						pos += stride;
						if (pos % out_width == 0)
						{
							pos += out_width;
						}
					}
				}
			}
		};

	}

}

#endif
