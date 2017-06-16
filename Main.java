package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class Main {

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

	/**
	 * Run a single experiment.
	 * 
	 * @throws Exception
	 */
	private static void run() throws Exception {
		
		// Load data
		Instances trainingData = loadData("ElectionsData_train.txt");
		
		// Train classifier
		SVMEval mySVM = new SVMEval();
		mySVM.chooseKernel(trainingData);
		Instances workingSet = mySVM.backwardsWrapper(trainingData, 0.05, 5);
		mySVM.buildClassifer(workingSet);
		
		// Test error
		System.out.println("Calculating average error on test set...");
		Instances testingData = loadData("ElectionsData_test.txt");
		Instances subsetOfFeatures = mySVM.removeNonSelectedFeatures(testingData);
		double avgError = mySVM.calcAvgError(subsetOfFeatures);
		System.out.println("Average error on test set is: " + avgError);
		
	}
	
	public static void main(String[] args) throws Exception {
		run();
	}
	
}
