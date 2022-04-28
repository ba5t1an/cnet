#ifndef CNET_VLOGGER_HPP
#define CNET_VLOGGER_HPP


#include <memory>
#include "cnet_layer.hpp"
#include "cnet_cv2eigen.hpp"

namespace Cnet
{
	class FeatureMapVisitor : public Visitor
	{
	private:
		std::string _location;
		bool _inplace;

	public:

		FeatureMapVisitor(std::string location, bool inplace = false) : Visitor()
		{
			this->_location = location;
			_inplace = inplace;
		}

		static void mkdir(std::string path)
		{
			if (!std::experimental::filesystem::exists(path))
			{
				std::experimental::filesystem::create_directories(path);
			}
		}

		void visit(Layer* layer) override
		{
			//Make sure that visitor is only applied to layers that produce feature maps
			if (layer->get_layer_type() == CONV2D_CC_LAYER || layer->get_layer_type() == CONV2D_LAYER || layer->get_layer_type() == CONV2D_T_LAYER || layer->get_layer_type() == INPUT_LAYER)
			{
				std::string outpath = _location;
				if (_current_iter == 0)
				{
					outpath += "/" + std::to_string(layer->get_layer_id()) + "_" + layer->get_layer_name();
					mkdir(outpath);
				}
				else
				{
					if (!_inplace)
					{
						outpath += "/" + std::to_string(_current_iter) + "_inter";
						mkdir(outpath);
						outpath += "/" + std::to_string(layer->get_layer_id()) + "_" + layer->get_layer_name();
						mkdir(outpath);
					}
					else
					{
						outpath += "/" + std::to_string(layer->get_layer_id()) + "_" + layer->get_layer_name();
						mkdir(outpath);
					}
				}
				for (unsigned int i = 0; i < layer->get_output()->rows(); ++i)
				{
					std::string out_image_path = outpath + "/ch" + std::to_string(i) + ".png";
					MatrixRm mat = layer->get_output()->row(i);
					save_image(mat, out_image_path);
				}
			}
		}
		~FeatureMapVisitor() = default;
	};


	class ShapeVisitor : public Visitor
	{
	public:
		ShapeVisitor() : Visitor()
		{}

		void visit(Layer* layer) override
		{
		    if (layer->get_layer_type() == FC_LAYER)
            {
                std::cout << layer->get_layer_name() << ": [id=" << layer->get_layer_id() << "][shape=( 1, 1, " << layer->get_output_size() << ")]" << "[act=" << std::to_string(layer->get_activation()->get_type()) << "]" << std::endl;
            } else
            {
                size_t dim = (size_t)sqrt(layer->get_output_size());
                std::string info = layer->get_layer_name() + ": [id=" + std::to_string(layer->get_layer_id()) + "][shape=(" + std::to_string(layer->get_output_channels()) + ", " + std::to_string(dim) + ", " + std::to_string(dim) + ")]";
				if (layer->get_activation())
				{
					info += +"[act=" + std::to_string(layer->get_activation()->get_type()) + "]";
				}
				std::cout << info << std::endl;
			}
		}
		~ShapeVisitor() = default;
	};
}
#endif // !CNET_VLOGGER_HPP
