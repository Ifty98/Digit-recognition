import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
	// This program trains a neural network model using digit data from two CSV datasets,
	// normalizes pixel values, and evaluates the model's accuracy on a separate test dataset.
	// It prints the final count of correct predictions for each digit label (0 to 9).
	public static void main(String[] args) {
		//file paths for datasets
		String dataSet1 = "cw2DataSet1.csv";
		String dataSet2 = "cw2DataSet2.csv";
		//array to store result counts for each label
		int[] results = {0,0,0,0,0,0,0,0,0,0};

		//read data from files into ArrayLists
		ArrayList<DigitData> digitDataList1 = readDataFromFile(dataSet1);
		ArrayList<DigitData> digitDataList2 = readDataFromFile(dataSet2);
	    
		//normalize pixel values in digit data
		for (int index = 0; index < digitDataList1.size(); index++) {
			digitDataList1.get(index).normalizePixels();
		}
		
		for (int index = 0; index < digitDataList2.size(); index++) {
			digitDataList2.get(index).normalizePixels();
		}
		//create a Neural Network model
		NeuralNetwork nnmodel = new NeuralNetwork(64, 20, 10);
		//train the model for a certain number of rounds
		for (int round = 0; round < 10; round++) {
			for (int index = 0; index < digitDataList1.size(); index++) {
				nnmodel.trainModel(digitDataList1.get(index).getPixels(), digitDataList1.get(index).getLabelMatrix());
			}
		}
		//test the model and update result counts
		for (int index = 0; index < digitDataList2.size(); index++) {
			double[][] predictions = nnmodel.testModel(digitDataList2.get(index).getPixels());
			int label = checkResults(predictions);
			for (int resIndex = 0; resIndex < results.length; resIndex++) {
				if (label == resIndex) {
					results[resIndex] += 1;
				}
			}
		}
		printResults(results);
	}

	//method to read data from a file and return a list of DigitData objects
	public static ArrayList<DigitData> readDataFromFile(String filePath) {
		ArrayList<DigitData> digitDataList = new ArrayList<>();

		try (Scanner scanner = new Scanner(new File(filePath))) {
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] elements = line.split(",");

				//parse the label (the last number in the line)
				int label = Integer.parseInt(elements[elements.length - 1]);

				//parse the pixels
				double[][] pixels = new double[64][1];
				int pixelIndex = 0;
				for (int index = 0; index < pixels.length; index++) {
						pixels[index][0] = Integer.parseInt(elements[pixelIndex]);
						pixelIndex++;
				}

				DigitData digitData = new DigitData(label, pixels);
				digitDataList.add(digitData);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return digitDataList;
	}
	 
	//method to check the predicted label based on the model's output
    public static int checkResults(double[][] predictions) {
    	int index = 0;
    	double max = Integer.MIN_VALUE;
    	for (int row = 0; row < predictions.length; row++) {
    		for (int col = 0; col < predictions[0].length; col++) {
    			if (max < predictions[row][col]) {
    				max = predictions[row][col];
    				index = row;
    			}
    		}
    	}
    	return index;
    }
    
    //method to print the final results
    public static void printResults(int[] results) {
    	System.out.println("Results: ");
    	System.out.println("Label 0 = " + results[0]);
    	System.out.println("Label 1 = " + results[1]);
    	System.out.println("Label 2 = " + results[2]);
    	System.out.println("Label 3 = " + results[3]);
    	System.out.println("Label 4 = " + results[4]);
    	System.out.println("Label 5 = " + results[5]);
    	System.out.println("Label 6 = " + results[6]);
    	System.out.println("Label 7 = " + results[7]);
    	System.out.println("Label 8 = " + results[8]);
    	System.out.println("Label 9 = " + results[9]);
    }
}
