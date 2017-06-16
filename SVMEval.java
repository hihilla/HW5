package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SVMEval {
	
	private static final int NUM_OF_FOLDS = 3;
	private static final int NUM_OF_KERNELS = 16;
	
	private SMO mySMO;
	private Kernel kernel;
	private ArrayList<Integer> removedIndexesInOrder;
	
	/**
	 * Instantiates an SVM classifier.
	 * 
	 */
	public SVMEval() {
		mySMO = new SMO();
		removedIndexesInOrder = new ArrayList<Integer>();
	}
	
	/**
	 * Builds the SVM classifier.
	 * 
	 * @param data
	 * @throws Exception 
	 */
	public void buildClassifer(Instances data) throws Exception {
		mySMO = new SMO();
		mySMO.setKernel(this.kernel);
		mySMO.buildClassifier(data);
	}
	
	/**
	 * Chooses the best kernel for the SVM classifier and sets it.
	 *  
	 * @param data
	 * @throws Exception 
	 */
	public void chooseKernel(Instances data) throws Exception {
		
		System.out.println("Calculating score for all kernels...");
		
		// Fill kernels array
		Kernel[] kernels = new Kernel[NUM_OF_KERNELS];
		for (int i=2; i<=4; i++) {
			PolyKernel pk = new PolyKernel();
			pk.setExponent(i);
			kernels[i-2] = pk;
		}
		for (int i=-10; i<=2; i++) {
			RBFKernel rbfk = new RBFKernel();
			rbfk.setGamma(Math.pow(2, i));
			kernels[i+13] = rbfk;
		}
		
		// Iterate over all the kernels
		double minXVE = Double.POSITIVE_INFINITY;
		int minKernelIndex = -1;
		for (int i=0; i<NUM_OF_KERNELS; i++) {
			
			// Load the data and randomize it
			Instances allTheData = new Instances(data);
			allTheData.randomize(new Random());
			
			// Calculate the cross validation error for this configuration
			double xve = calcCrossValidationError(allTheData, kernels[i]);
			System.out.printf("\tFor i=%d the xve is %f\n", i, xve);
			
			// Save the lowest error parameters
			if (xve < minXVE) {
				minXVE = xve;
				minKernelIndex = i;
			}
			
		}
		
		// Save the selected kernel function
		this.kernel = kernels[minKernelIndex];
		System.out.println("Choosing kernel #" + minKernelIndex);
		
		// Set the best kernel
		mySMO = new SMO();
		mySMO.setKernel(kernel);
		
	}
	
	/**
	 * Performs the backwards wrapper feature selection algorithm.
	 * 
	 * @param data
	 * @param threshold
	 * @param minNumAttributes
	 * @throws Exception 
	 */
	public Instances backwardsWrapper(Instances data, double threshold, int minNumAttributes) throws Exception {
		
		System.out.println("Calculating the best attributes...");
		
		// Initialize variables
		double error_diff = 0;
		double original_error = calcCrossValidationError(data, this.kernel);
		Instances allTheData = new Instances(data);
		boolean thresholdReached = false;
		
		System.out.println("Original xve is " + original_error);
		
		do {
			
			System.out.println("*** NEW ROUND ***");
			
			// Remove attribute #1 and check error
			int i_minimal = 1;
			Instances allButFeatureOne = removeAttributeFromDataset(allTheData, 1);
			double minimal_error = calcCrossValidationError(allButFeatureOne, this.kernel);
			
			// Iterate over all remaining attributes and choose minimal error
			for (int i=2; i<allTheData.numAttributes(); i++) {
				
				Instances allButFeatureI = removeAttributeFromDataset(allTheData, i);
				double new_error = calcCrossValidationError(allButFeatureI, this.kernel);
				if (new_error < minimal_error) {
					minimal_error = new_error;
					i_minimal = i;
				}
				
			}
			
			// Remove selected attribute if diff is below threshold, stop otherwise
			error_diff = minimal_error - original_error;
			if (error_diff < threshold) {
				allTheData = removeAttributeFromDataset(allTheData, i_minimal);
				this.removedIndexesInOrder.add(i_minimal);
				System.out.printf("Attribute #%d removed\n", i_minimal);
			} else {
				thresholdReached = true;
			}
			
			System.out.printf("error_diff = %f, curr_error = %f, num of attributes = %d\n", error_diff, minimal_error, allTheData.numAttributes() - 1);
			
		} while (allTheData.numAttributes() - 1 > minNumAttributes && !thresholdReached);
			
		System.out.println("backwardsWrapper is done!");
		
		return allTheData;
		
	}
	
	/**
	 * Removes the index'th attribute from the given dataset.
	 * 
	 * @param allTheData
	 * @param index
	 * @return
	 * @throws Exception
	 */
	private Instances removeAttributeFromDataset(Instances allTheData, int index) throws Exception {
		Remove remove = new Remove();
		remove.setInputFormat(allTheData);
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = Integer.toString(index + 1);
		remove.setOptions(options);
		Instances workingSet = Filter.useFilter(allTheData, remove);
		return workingSet;
	}
	
	/**
	 * Calculates the average error of the classifier on the given dataset.
	 * 
	 * @param data
	 * @return
	 * @throws Exception 
	 */
	public double calcAvgError(Instances data) throws Exception {
		double count = 0;
		for (int i=0; i<data.numInstances(); i++) {
			Instance ins = data.instance(i);
			double realClassValue = ins.classValue();
			double predClassValue = mySMO.classifyInstance(ins);
			if (realClassValue != predClassValue) {
				count++;
			}
		}
		return count / data.numInstances();
	}
	
	/**
	 * Calculate the cross validation error of the classifier using the given dataset and kernel.
	 * 
	 * @param data
	 * @return
	 * @throws Exception 
	 */
	public double calcCrossValidationError(Instances data, Kernel kernel) throws Exception {

		double xverror = 0;
		
		for (int i=0; i<NUM_OF_FOLDS; i++) {
			
			// Calculate the start and end indexes of the testing data
			int foldSize = data.numInstances() / NUM_OF_FOLDS;
			int startIndex = i * foldSize;
			int endIndex = startIndex + foldSize - 1;
			
			// Last round - grab the remainder too
			if (i == NUM_OF_FOLDS - 1) {
				endIndex = data.numInstances() - 1;
			}
			
			// Divide the data into training and testing groups
			Instances trainingData = new Instances(data, data.numInstances());
			Instances testingData = new Instances(data, data.numInstances());
			for (int j=0; j<data.numInstances(); j++) {
				if (j >= startIndex && j <= endIndex) {
					testingData.add(data.instance(j));
				} else {
					trainingData.add(data.instance(j));
				}
			}
			
			// Train the classifier
			try {
				mySMO = new SMO();
				mySMO.setKernel(kernel);
				mySMO.buildClassifier(trainingData);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			// Calculate the average error for this fold
			xverror += calcAvgError(testingData);
		}
		
		return xverror / NUM_OF_FOLDS;
		
	}
	
	/**
	 * Removes non selected features from the dataset.
	 * 
	 * @param data
	 * @return
	 * @throws Exception 
	 */
	public Instances removeNonSelectedFeatures(Instances data) throws Exception {
		System.out.println("removeNonSelectedFeatures:");
		Instances newInstances = new Instances(data);
		for (int i=0; i<this.removedIndexesInOrder.size(); i++) {
			newInstances = removeAttributeFromDataset(newInstances, this.removedIndexesInOrder.get(i));
			System.out.println("\tremoved attribute #" + this.removedIndexesInOrder.get(i));
		}
		System.out.println("removeNonSelectedFeatures is done");
		return newInstances;
	}

	/**
	 * Reads data from file.
	 * 
	 * @param filename
	 * @return
	 */
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * 
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(0);
		return data;
	}
	
}
