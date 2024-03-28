// NeuralNetwork class represents a simple feedforward neural network with one hidden layer,
// trained using backpropagation. It has methods to initialize the network with random weights,
// train the model, and perform forward and backward passes for predictions and weight updates.
// The network uses ReLU activation for the hidden layer and softmax activation for the output layer.
// Training is done using stochastic gradient descent with a specified learning rate.
public class NeuralNetwork {
    //instance variable
	private int inputSize;
	private int hiddenSize;
	private int outputSize;
	private double[][] hiddenLayerWeights;
	private double[][] outputLayerWeights;
	private double[][] hiddenLayerBias;
	private double[][] outputLayerBias;
	private double learningRate = 0.001;

	//constructor to initialize the neural network with specified sizes
	public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;
		//initialize biases and weights with random values
		this.hiddenLayerBias = generateBias(this.hiddenSize);
		this.outputLayerBias = generateBias(this.outputSize);
		this.hiddenLayerWeights = generateRandomHiddenWeights(this.hiddenSize, this.inputSize);
		this.outputLayerWeights = generateRandomOutputWeights(this.outputSize, this.hiddenSize);
	}

	//method to train the neural network using backpropagation
	public void trainModel(double[][] input, double[][] expectedOutput) {
		//forward propagation
		//compute hidden layer input
		double[][] hiddenLayerProduct = multiplyMatrices(this.hiddenLayerWeights, input);
		double[][] hiddenLayerInput = sumMatrices(hiddenLayerProduct, this.hiddenLayerBias);
		//apply activation function (ReLU) to get hidden layer output
		double[][] hiddenLayerOutput = relu(hiddenLayerInput);
       
		//compute output layer input
		double[][] outputLayerProduct = multiplyMatrices(this.outputLayerWeights, hiddenLayerOutput);
		double[][] outputLayerInput = sumMatrices(outputLayerProduct, this.outputLayerBias);
		//apply activation function (Softmax) to get predicted output
		double[][] predictedOutput = softmax(outputLayerInput);
		
		//backward propagation
		//compute error between predicted and expected output
		double[][] error = differenceOfMatrices(predictedOutput, expectedOutput);
		
		//update weights and biases for the output layer
		this.outputLayerWeights = sumMatrices(this.outputLayerWeights, 
				multiplyByNumber(multiplyMatrices(error, transpose(hiddenLayerOutput)) , -this.learningRate));
		this.outputLayerBias = sumMatrices(this.outputLayerBias, multiplyByNumber(error , -this.learningRate));
		
		//compute error for the hidden layer
		double[][] hiddenError = multiplyMatrices(transpose(this.outputLayerWeights), error);
		//apply derivative of activation function (ReLU) to hidden layer error
		double[][] reluDeriv = reluDerivative(hiddenLayerInput);
		hiddenError = multiplyMatricesElementWise(hiddenError, reluDeriv);

		//update weights and biases for the hidden layer
		this.hiddenLayerWeights = sumMatrices(this.hiddenLayerWeights, 
				multiplyByNumber(multiplyMatrices(hiddenError, transpose(input)) , -this.learningRate));
		this.hiddenLayerBias = sumMatrices(this.hiddenLayerBias, multiplyByNumber(hiddenError , -this.learningRate));
		
	}

	//method to test the neural network on new input data
	public double[][] testModel(double[][] input) {
		//forward pass
		//compute hidden layer input
		double[][] hiddenLayerProduct = multiplyMatrices(this.hiddenLayerWeights, input);
		double[][] hiddenLayerInput = sumMatrices(hiddenLayerProduct, this.hiddenLayerBias);
		//apply activation function (ReLU) to get hidden layer output
		double[][] hiddenLayerOutput = relu(hiddenLayerInput);

		//compute output layer input
		double[][] outputLayerProduct = multiplyMatrices(this.outputLayerWeights, hiddenLayerOutput);
		double[][] outputLayerInput = sumMatrices(outputLayerProduct, this.outputLayerBias);
		//apply activation function (Softmax) to get predicted output
		double[][] predictedOutput = softmax(outputLayerInput);
		
		//return the predicted output
		return predictedOutput;
	}

	//generate random weights for the hidden layer with a specific size
	public double[][] generateRandomHiddenWeights(int size1, int size2) {

		double[][] randomWeights = new double[size1][size2];

		// Initialize weights with random values using He initialization
		for (int row = 0; row < size1; row++) {
			for (int col = 0; col < size2; col++) {
				randomWeights[row][col] = Math.random() - 0.5 * Math.sqrt(2.0 / size1);
			}
		}

		return randomWeights;
	}

	//generate random weights for the output layer with a specific size
	public double[][] generateRandomOutputWeights(int size1, int size2) {

		double[][] randomWeights = new double[size1][size2];
		//initialize weights with random values
		for (int row = 0; row < size1; row++) {
			for (int col = 0; col < size2; col++) {
				randomWeights[row][col] = Math.random() - 0.5;
			}
		}

		return randomWeights;
	}

	//generate random bias values for a layer with a specific size
	public double[][] generateBias(int size) {

		double[][] randomWeights = new double[size][1];
		//initialize bias values with random values
		for (int row = 0; row < size; row++) {
			for (int col = 0; col < 1; col++) {
				randomWeights[row][col] = Math.random() - 0.5;
			}
		}

		return randomWeights;

	}
	
	//apply softmax operation to a 2D matrix
	public double[][] softmax(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] result = new double[rows][cols];

        //flatten the matrix into a 1D array
        double[] flattenedMatrix = new double[rows * cols];
        int index = 0;
        for (int matrixRow = 0; matrixRow < rows; matrixRow++) {
            for (int matrixCol = 0; matrixCol < cols; matrixCol++) {
                flattenedMatrix[index++] = matrix[matrixRow][matrixCol];
            }
        }

        //apply softmax to the flattened array
        double[] softmaxResult = softmax(flattenedMatrix);

        //reshape the softmax result back to a 2D array
        index = 0;
        for (int matrixRow = 0; matrixRow < rows; matrixRow++) {
            for (int matrixCol = 0; matrixCol < cols; matrixCol++) {
                result[matrixRow][matrixCol] = softmaxResult[index++];
            }
        }

        return result;
    }
	
	//apply softmax operation to a 1D array
	public static double[] softmax(double[] array) {
	    int length = array.length;

	    //find the maximum value in the array
	    double maxVal = Double.NEGATIVE_INFINITY;
	    for (double value : array) {
	        maxVal = Math.max(maxVal, value);
	    }

	    double expSum = 0.0;
	    double[] result = new double[length];
	    //calculate the numerator and accumulate the denominator
	    for (int index = 0; index < length; index++) {
	        result[index] = Math.exp(array[index] - maxVal);
	        expSum += result[index];
	    }

	    //normalize by the accumulated denominator
	    for (int index = 0; index < length; index++) {
	        if (expSum != 0.0) {
	            result[index] /= expSum;
	        } else {
	            result[index] = 1.0 / length;
	        }
	    }

	    return result;
	}

    //ReLU function for a 2D array
    public double[][] relu(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        //apply ReLU activation element-wise to each element in the matrix
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result[row][col] = Math.max(0, matrix[row][col]);
            }
        }

        return result;
    }

    //ReLU derivative function for a 2D array
    public double[][] reluDerivative(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        //calculate the derivative of ReLU activation element-wise
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result[row][col] = matrix[row][col] > 0 ? 1 : 0;
            }
        }

        return result;
    }

    //matrix multiplication function for two 2D arrays
	public double[][] multiplyMatrices(double[][] matrix1, double[][] matrix2) {
		int rows1 = matrix1.length;
		int cols1 = matrix1[0].length;
		int rows2 = matrix2.length;
		int cols2 = matrix2[0].length;

		//check if the matrices have compatible dimensions for multiplication
		if (cols1 != rows2) {
			throw new IllegalArgumentException("Invalid matrix dimensions for multiplication");
		}

		double[][] result = new double[rows1][cols2];

		//perform matrix multiplication using nested loops
		for (int row = 0; row < rows1; row++) {
			for (int col1 = 0; col1 < cols2; col1++) {
				double sum = 0;
				for (int col2 = 0; col2 < cols1; col2++) {
					sum += matrix1[row][col2] * matrix2[col2][col1];
				}
				result[row][col1] = sum;
			}
		}

		return result;
	}

	//element wise multiplication function for two 2D arrays
	public static double[][] multiplyMatricesElementWise(double[][] matrix1, double[][] matrix2) {
		int rows = matrix1.length;
		int cols = matrix1[0].length;

		//check if the matrices have equal dimensions for element-wise multiplication
		if (rows != matrix2.length || cols != matrix2[0].length) {
			throw new IllegalArgumentException("Matrices must have equal dimensions for element-wise multiplication");
		}

		double[][] result = new double[rows][cols];

		//perform element-wise multiplication using nested loops
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = matrix1[row][col] * matrix2[row][col];
			}
		}

		return result;
	}

	//multiplication of 2D array by a number
	public double[][] multiplyByNumber(double[][] matrix, double number) {
		int rows = matrix.length;
		int cols = matrix[0].length;
		double[][] result = new double[rows][cols];

		//multiply each element of the matrix by the given number
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = matrix[row][col] * number;
			}
		}

		return result;
	}

	//matrix addition function for two 2D arrays
	public double[][] sumMatrices(double[][] matrix1, double[][] matrix2) {
		int rows1 = matrix1.length;
		int cols1 = matrix1[0].length;
		int rows2 = matrix2.length;
		int cols2 = matrix2[0].length;

		//check if the matrices have the same dimensions for addition
		if (rows1 != rows2 || cols1 != cols2) {
			throw new IllegalArgumentException("Shape Mismatch");
		}

		double[][] result = new double[rows1][cols1];

		//perform element-wise addition using nested loops
		for (int row = 0; row < rows1; row++) {
			for (int col = 0; col < cols1; col++) {
				result[row][col] = matrix1[row][col] + matrix2[row][col];
			}
		}

		return result;
	}

	//addition of a matrix by a number
	public double[][] sumByNumber(double[][] matrix, double number) {
		int rows = matrix.length;
		int cols = matrix[0].length;

		double[][] result = new double[rows][cols];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[row][col] = matrix[row][col] + number;
			}
		}

		return result;
	}

	//element wise subtraction of two matrices
	public double[][] differenceOfMatrices(double[][] matrix1, double[][] matrix2) {

		int rows1 = matrix1.length;
		int cols1 = matrix1[0].length;
		int rows2 = matrix2.length;
		int cols2 = matrix2[0].length;

		if (rows1 != rows2 || cols1 != cols2) {
			throw new IllegalArgumentException("Shape Mismatch");
		}

		double[][] result = new double[rows1][cols1];

		for (int row = 0; row < rows1; row++) {
			for (int col = 0; col < cols1; col++) {
				result[row][col] = matrix1[row][col] - matrix2[row][col];
			}
		}

		return result;

	}
	
	//subtract a number from all elements of a matrix
	public double[][] subtractNumberFromMatrix(double[][] matrix, double number) {
	    int rows = matrix.length;
	    int cols = matrix[0].length;

	    double[][] result = new double[rows][cols];

	    for (int row = 0; row < rows; row++) {
	        for (int col = 0; col < cols; col++) {
	            result[row][col] = matrix[row][col] - number;
	        }
	    }

	    return result;
	}

	//transpose a matrix
	public double[][] transpose(double[][] matrix) {
		int rows = matrix.length;
		int cols = matrix[0].length;

		double[][] result = new double[cols][rows];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				result[col][row] = matrix[row][col];
			}
		}

		return result;
	}
}
