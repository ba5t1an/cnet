/**
Implementation of several parser functions.

@file cnet_parser.hpp
@author Bastian Schoettle EN RC PREC
*/

#ifndef CNET_PARSER_HPP
#define CNET_PARSER_HPP

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <math.h>
#include "cnet_dataset.hpp"
#include "cnet_common.hpp"

namespace Cnet
{


	inline MatrixRm vec_to_eigen(std::vector<float>& data)
	{
		MatrixRm mat = MatrixRm::Zero(1, data.size());
		for (size_t i = 0; i < data.size(); ++i)
		{
			mat(0, i) = data[i];
		}
		return mat;
	}

	inline int reverse_int(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}

	inline void read_mnist_images(std::string filename, std::vector<std::vector<float> > &vec)
	{
		std::ifstream file(filename, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = reverse_int(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = reverse_int(number_of_images);
			file.read((char*)&n_rows, sizeof(n_rows));
			n_rows = reverse_int(n_rows);
			file.read((char*)&n_cols, sizeof(n_cols));
			n_cols = reverse_int(n_cols);
			for (int i = 0; i < number_of_images; ++i)
			{
				std::vector<float> tp;
				for (int r = 0; r < n_rows; ++r)
				{
					for (int c = 0; c < n_cols; ++c)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						tp.push_back((float)temp);
					}
				}
				vec.push_back(tp);
			}
		}
	}

	inline void read_mnist_labels(std::string filename, std::vector<float> &vec)
	{
		std::ifstream file(filename, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = reverse_int(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = reverse_int(number_of_images);
			if ((int)vec.size() < number_of_images)
			{
				vec.resize(number_of_images);
			}
			for (int i = 0; i < number_of_images; ++i)
			{
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				vec[i] = (float)temp;
			}
		}
	}

	inline void read_mnist(InMemoryDataset& dataset, std::string images_path)
	{
		std::vector<std::vector<float> > images;
		read_mnist_images(images_path, images);
		size_t num_images = images.size();
		for (size_t i = 0; i < num_images; i++)
		{
			dataset.add_sample(Entry(vec_to_eigen(images[i])), Entry(vec_to_eigen(images[i])));
		}
	}

	inline int read_mnist(InMemoryDataset& dataset, std::string images_path, std::string labels_path, size_t num_items = 0)
	{
		std::cout << "INFO:: Reading data from (" << images_path << ", " << labels_path << ")"<< std::endl;
		if (!file_exists(images_path) || !file_exists(labels_path))
		{
			return -1;
		}
		std::vector<std::vector<float> > images;
		read_mnist_images(images_path, images);
		std::vector<float> labels;
		read_mnist_labels(labels_path, labels);
		size_t num_images = images.size();
		for (size_t i = 0; i < num_images; ++i)
		{
			dataset.add_sample(Entry(vec_to_eigen(images[i]), images_path), labels[i]);
			if (i+1 % 10000 == 0)
			{
				std::cout << "INFO:: Added " << std::to_string(i) << " of " <<  std::to_string(num_images) << " items" << std::endl;
			}
			if (num_items > 0 && i+1 == num_items)
			{
				return 0;
			}
		}
		return 0;
	}


	/*
	Method to read grayscale images in caffe format, e.g. class-label <path_to_image> ...
	*/
	inline int read_caffe_format(OnlineDataset& data, std::string file_name, size_t num_items = 0)
	{
		std::ifstream file;
		file.open(file_name);
		if (!file)
		{
			std::cout << "ERROR: Unable to read file" << std::endl;
			return 0;
		}
		std::string line;
		size_t cnt = 0;
		while (std::getline(file, line))
		{
			std::string label(line.substr(0, 1));
			std::string path = line.substr(2);
			data.add_sample(path, label);
			if(++cnt == num_items)
            {
			    return 1;
            }
		}
		return 1;
	}


	/*
	Method to read grayscale images in caffe format, e.g. class-label <path_to_image> ...
	*/
	inline int read_caffe_format(InMemoryDataset& data, std::string file_name, size_t num_items = 0)
	{
		std::cout << "INFO:: (Parser) Reading data from " << file_name <<  std::endl;

		std::ifstream file;
		file.open(file_name);
		if (!file)
		{
			std::cout << "ERROR: Unable to read file" << std::endl;
			return 0;
		}
		std::string line;
		size_t cnt = 0;
		while (std::getline(file, line))
		{
			std::string label(line.substr(0, 1));
			std::string path = line.substr(2);
			MatrixRm input;
			load_image(input, path);
			data.add_sample(Entry(input, path), std::stoi(label));
            if(++cnt == num_items)
            {
                return 1;
            }
		}
		//file.close();
		return 1;
	}


	/*
	Method to read grayscale images in sementation format, e.g. class-label <path_to_image> ...
	*/
	inline int read_seg_format(OnlineDataset& data, std::string file_name, char separator = ' ', size_t num_items = 0)
	{
		if (!file_exists(file_name))
		{
			return 0;
		}
		std::ifstream file;
		file.open(file_name);
		if (!file)
		{
			std::cout << "ERROR: Unable to read file" << std::endl;
			return 0;
		}
		size_t item_cnt = 0;
		std::string line;
		while (std::getline(file, line))
		{

			std::stringstream ss(line);
			std::string item;
			std::string input;
			std::string label;
			size_t cnt = 0;
			while (std::getline(ss, item, separator))
			{
				if (cnt == 0)
				{
					input = item;
				}
				else if (cnt == 1)
				{
					label = item;
				}
				++cnt;
			}
			data.add_sample(input, label);
			++item_cnt;
			if (num_items > 0 && item_cnt == num_items)
			{
				return 1;
			}
		}
		return 1;
	}


	/*
	Method to read grayscale images in sementation format, e.g. class-label <path_to_image> ...
	*/
	inline int read_seg_format(InMemoryDataset& data, std::string file_name, char separator = ' ', size_t num_items = 0)
	{
		if (!file_exists(file_name))
		{
			return 0;
		}
		std::ifstream file;
		file.open(file_name);
		if (!file)
		{
			std::cout << "ERROR: Unable to read file" << std::endl;
			return 0;
		}
		size_t item_cnt = 0;
		std::string line;
		while (std::getline(file, line))
		{

			std::stringstream ss(line);
			std::string item;
			std::string input;
			std::string label;
			size_t cnt = 0;
			while (std::getline(ss, item, separator))
			{
				if (cnt == 0)
				{
					input = item;
				}
				else if (cnt == 1)
				{
					label = item;
				}
				++cnt;
			}
			MatrixRm input_img;
			MatrixRm label_img;
			load_image(input_img, input);
			load_image(label_img, label);
			data.add_sample(Entry(input_img, input), Entry(label_img, label));
			++item_cnt;
			if (num_items > 0 && item_cnt == num_items)
			{
				return 1;
			}
		}
		return 1;
	}


}

#endif


