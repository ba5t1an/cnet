# cnet
Cnet is a lightweight, header-only framework to train and use Convolutional Neural Networks and MLPs. See examples/examples.cpp on how to use it.

Currently, the framework supports the following layer types:

    - Conv2D
    - Conv2DTranspose
    - FullyConnected
    - CropAndConcat
    - MaxPool2D

Training can be performed in parallel on CPU. 

## Training on MNIST

The training on MNIST is fairly simple and should demonstrate how to use the framework. First, you'll have to get the database from http://yann.lecun.com/exdb/mnist/. Afterwards, cnet offers the required functions to read the format.	

    Cnet::DatasetParams params;
	params.encode_one_hot = true;
	params.num_classes = 10;
	std::shared_ptr<Cnet::Pipeline> pipeline = std::make_shared<Cnet::Pipeline>();
	pipeline->add_operation(std::unique_ptr<Cnet::ScaleOp>(new Cnet::ScaleOp(255)));
	Cnet::InMemoryDataset train_data(params, pipeline);
	Cnet::read_mnist(train_data, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	Cnet::InMemoryDataset val_data(params, pipeline);
	Cnet::read_mnist(val_data, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	std::shared_ptr<Cnet::Layer> input = Cnet::InputLayer::create(28, 1);
	std::shared_ptr<Cnet::Layer> conv2d_1 = Cnet::Conv2dLayer::create(input, 2, 3,1,Cnet::VALID);
	std::shared_ptr<Cnet::Layer> conv2d_2 = Cnet::Conv2dLayer::create(conv2d_1, 2, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> pool_1 = Cnet::MaxPool2dLayer::create(conv2d_2, 2, 2);
	std::shared_ptr<Cnet::Layer> conv2d_3 = Cnet::Conv2dLayer::create(pool_1, 4, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> conv2d_4 = Cnet::Conv2dLayer::create(conv2d_3, 4, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> pool_2 = Cnet::MaxPool2dLayer::create(conv2d_4, 2, 2);
	std::shared_ptr<Cnet::Layer> fc_layer_1 = Cnet::FcLayer::create(pool_2, 128);
	std::shared_ptr<Cnet::Layer> fc_layer_2 = Cnet::FcLayer::create(fc_layer_1, 10, Cnet::SOFTMAX);
	Cnet::Graph graph(fc_layer_2);
	std::shared_ptr<Cnet::Evaluator> eval = std::make_shared<Cnet::ClassificationEvaluator>();
	Cnet::Estimator estimator(graph, eval,std::unique_ptr<Cnet::Loss>(new Cnet::CrossEntropyLoss), std::unique_ptr<Cnet::Solver>(new Cnet::AdamSolver(0.001)));
	estimator.train(train_data, val_data, 8, 100000, 100, 0, 1000);
	std::cout << "INFO:: (Main) Running test after training..." << std::endl;
	float val = eval->test(graph, val_data);
	std::cout << "INFO:: (Main) Model scored " << val << std::endl;

The resulting accuracy should be around 98% after training.





