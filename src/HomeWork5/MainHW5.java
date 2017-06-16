package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

public class MainHW5 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static ResultBestKernel findBestKernel(Instances instances){
		
		int division = 5;
		Instances testData = new Instances(instances, instances.numInstances());
		Instances trainData = new Instances(instances, instances.numInstances());
		String bestKernel = "";
		double bestKerVal = -1;
		double bestKernelResults = (double)Integer.MIN_VALUE;
		
		for (int i = 0; i < instances.numInstances(); i++) {
			if (i % division == 0){
				testData.add(instances.instance(i));
			} else {
				trainData.add(instances.instance(i));
			}
		}
		
		int[] polynomialKernel = {2, 3, 4};
		double[] RBFKernel = {1.0 / 100, 1.0 / 10, 1.0};
		
		for (int kernelValue : polynomialKernel) {
			PolyKernel polyKer = new PolyKernel();
			polyKer.setExponent(kernelValue);
			SVM svm = new SVM();
			svm.setKernel(polyKer);
			try {
				svm.buildClassifier(trainData);
				int[] confusion = svm.calcConfusion(testData);
				double[] confusionRates = svm.calcConfRates(confusion);
				System.out.println("For PolyKernel with degree "+ kernelValue +" the rates are:");
				System.out.println("TPR = " + confusionRates[0]);
				System.out.println("TPR = " + confusionRates[1]);
				if (bestKernelResults > (confusionRates[0] - confusionRates[1])){
					bestKerVal = kernelValue;
					bestKernel = "Poly";
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		}
		
		for (double kernelValue : RBFKernel) {
			RBFKernel RBFker = new RBFKernel();
			RBFker.setGamma(kernelValue);
			SVM svm = new SVM();
			svm.setKernel(RBFker);
			try {
				svm.buildClassifier(trainData);
				int[] confusion = svm.calcConfusion(testData);
				double[] confusionRates = svm.calcConfRates(confusion);
				System.out.println("For RBFKernel with gamma "+ kernelValue +" the rates are:");
				System.out.println("TPR = " + confusionRates[0]);
				System.out.println("TPR = " + confusionRates[1]);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		return new ResultBestKernel(bestKernel, bestKerVal);
	}
	
	public static void main(String[] args) throws Exception {
		
		Instances data = loadData("cancer.txt");
		
		
		
	}
	
	
}
class ResultBestKernel {
	String kerType;
	double kerValue;
	public ResultBestKernel(String kerType, double kerValue) {
		this.kerType = kerType;
		this.kerValue = kerValue;
	}
}
