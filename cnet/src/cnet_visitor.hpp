#ifndef CNET_Visitor_HPP
#define CNET_Visitor_HPP


#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
#include <experimental/filesystem>
#include "cnet_layer.hpp"

namespace Cnet
{
	class Layer; 

	class Visitor
	{
	protected:
		friend class Layer;
		size_t _current_iter;

	public:

		Visitor()
		{
			_current_iter = 0;
		}

		virtual void visit(Layer* layer) = 0;

		void set_current_iter(size_t current_iter)
		{
			_current_iter = current_iter;
		}
	};
}

#endif // ! CNET_Visitor_HPP

