# cnet
Cnet is a lightweight, header-only framework to train and use Convolutional Neural Networks and MLPs. See examples/examples.cpp on how to use it.

Currently, the framework supports the following layer types:

    - Conv2D
    - Conv2DTranspose
    - FullyConnected
    - CropAndConcat
    - MaxPool2D

Training can be performed in parallel on CPU.

The following 3rd parties are required to use this project:

    - Eigen (>= 3.x)
    - Boost (>= 1.6)
    - OpenCV (>= 3.x)


Note: The conversation between Eigen and OpenCV and vise versa uses a header file by Eugene Khvedchenya. This can be retrieved via https://gist.github.com/BloodAxe/c94d65d5977fb1d3e53f and must be renamed appropriately. It's also possible to use OpenCV's build-in functions for this task. May update it soon. 

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
	/*
	Model definition:
		-	InputLayer	:	Images of dimensions 28x28x1 
		-	Conv2DLayer	:	Filters of dimensions 2x3x3, stride 1 and valid padding
		-	Conv2DLayer	:	Filters of dimensions 2x3x3, stride 1 and valid padding
		-	MaxPool2D	:	Pooling with 2x2 windows and stride 2 
		-	Conv2DLayer	:	Filters of dimensions 4x3x3, stride 1 and valid padding
		-	Conv2DLayer	:	Filters of dimensions 4x3x3, stride 1 and valid padding
		-	MaxPool2D	:	Pooling with 2x2 windows and stride 2
		-	FcLayer		:	Fully connected layer with 128 output nodes, input nodes depend on previous layer
		-	FcLayer		:	Fully connected layer with 10 output nodes, input nodes depend on previous layer	 
	*/
	std::shared_ptr<Cnet::Layer> input = Cnet::InputLayer::create(28, 1);
	std::shared_ptr<Cnet::Layer> conv2d_1 = Cnet::Conv2dLayer::create(input, 2, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> conv2d_2 = Cnet::Conv2dLayer::create(conv2d_1, 2, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> pool_1 = Cnet::MaxPool2dLayer::create(conv2d_2, 2, 2);
	std::shared_ptr<Cnet::Layer> conv2d_3 = Cnet::Conv2dLayer::create(pool_1, 4, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> conv2d_4 = Cnet::Conv2dLayer::create(conv2d_3, 4, 3, 1, Cnet::VALID);
	std::shared_ptr<Cnet::Layer> pool_2 = Cnet::MaxPool2dLayer::create(conv2d_4, 2, 2);
	std::shared_ptr<Cnet::Layer> fc_layer_1 = Cnet::FcLayer::create(pool_2, 128);
	std::shared_ptr<Cnet::Layer> fc_layer_2 = Cnet::FcLayer::create(fc_layer_1, 10, Cnet::SOFTMAX);
	Cnet::Graph graph(fc_layer_2);
	/*
	Create evaluator with accuracy metric (https://en.wikipedia.org/wiki/Accuracy_and_precision)
	*/
	std::shared_ptr<Cnet::Evaluator> eval = std::make_shared<Cnet::ClassificationEvaluator>();
	/*
	Defines an estimator to train the model with the following settings:
		-	CrossEntropyLoss (https://en.wikipedia.org/wiki/Cross_entropy)
		-	AdamSolver with learning rate of .001 (https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)
	*/
	Cnet::Estimator estimator(graph, eval,std::unique_ptr<Cnet::Loss>(new Cnet::CrossEntropyLoss), std::unique_ptr<Cnet::Solver>(new Cnet::AdamSolver(0.001)));
	/*
	Trains the model with the following parameters:
		-	train_data		:	dataset with training data
		-	val_data		: 	dataset with validation data
		-	batch_size		: 	8
		-	max_iter		: 	max # of batches for the whole training
		-	display_iter	: 	display loss after every nth # of batch
		-	save_iter		:	save model every nth # of batches
		-	test_iter		:	test model every nth # of batches
	*/
	estimator.train(train_data, val_data, 8, 100000, 100, 0, 1000);
	std::cout << "INFO:: (Main) Running test after training..." << std::endl;
	/*
	Test model using accuracy metric (https://en.wikipedia.org/wiki/Accuracy_and_precision)
	*/
	float val = eval->test(graph, val_data);
	std::cout << "INFO:: (Main) Model scored " << val << std::endl;

The resulting accuracy should be around 98% after training.





