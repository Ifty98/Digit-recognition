//digitData class represents a digit with its label and pixel data
public class DigitData {
	private int label;//label of the digit
	private double[][] labelMatrix = new double[10][1];//one-hot encoded label matrix
	private double[][] pixels;//pixel data representing the digit image
	
	//constructor for initializing a DigitData object with label and pixels
	public DigitData(int label, double[][] pixels) {
		this.label = label;
		this.pixels = pixels;
		//initialize the one-hot encoded label matrix
		for (int index = 0; index < this.labelMatrix.length; index++) {
            if (index == label) {
                this.labelMatrix[index][0] = 1.0; //set the corresponding label to 1
            } else {
                this.labelMatrix[index][0] = 0.0; //set other labels to 0
            }
        }
	}
	
	// Getters
	public int getLabel() {
		return this.label;
	}
	
	public double[][] getPixels() {
		return this.pixels;
	}
	
	public double[][] getLabelMatrix() {
		return this.labelMatrix;
	}
	
	// Normalizes the pixel values to a range between 0 and 1
	public void normalizePixels() {
		for (int row = 0; row < pixels.length; row++) {
	        for (int col = 0; col < pixels[row].length; col++) {
	            pixels[row][col] = pixels[row][col] / 16.0;
	        }
	    }
	}
}
