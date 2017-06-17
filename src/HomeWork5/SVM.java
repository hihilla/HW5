package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instance;
import weka.core.Instances;

public class SVM {
	public SMO m_smo;

	public SVM() {
		this.m_smo = new SMO();
	}
	
	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}
	
	/**
	 * Calculate the TP, FP, TN, FN for a given instances object.
	 * Definitions:
	 * recurrence-events is the 0.0 class and will be the NEGATIVE class
	 * no-recurrence-events is the 1.0 class and will be the POSITIVE class
	 * @param instances
	 * @return int array of size 4 in this order [TP, FP, TN, FN].
	 */
	public int[] calcConfusion(Instances instances) throws Exception{
		int truePositive = 0; // prediction positive and condition positive 
		int falsePositive = 0; // prediction positive and condition negative
		int falseNegative = 0; // prediction negative and condition positive
		int trueNegative = 0; // prediction negative and condition negative

		// count population
		for (Instance instance : instances) {
			boolean conditionPositive = instance.classValue() == 1;
			boolean predictionPositive = this.m_smo.classifyInstance(instance) == 1;
			if (predictionPositive && conditionPositive) {
				truePositive++;
			} else if (predictionPositive && !conditionPositive) {
				falsePositive++;
			} else if (!predictionPositive && conditionPositive) {
				falseNegative++;
			} else if (!predictionPositive && !conditionPositive) {
				trueNegative++;
			}
		}
		
		int[] confusion = { truePositive, falsePositive, trueNegative, falseNegative }; 
		return confusion;
	}
	
	/**
	 * Calculate the TPR and FPR for the given cunfusion values.
	 * @param confusion - int array of size 4 in this order [TP, FP, TN, FN].
	 * @return int array of size 2 in this order [TPR, FPR].
	 */
	public double[] calcConfRates(int[] confusion) {
		double TPR = confusion[0] / ((double) confusion[0] + confusion[3]);
		double FPR = confusion[1] / ((double) confusion[1] + confusion[2]);
		double[] confRate = { TPR, FPR };
		return confRate;
	}
	
	/**
	 * Setting the Weka SMO classifier kernel
	 * @param kernel
	 */
	public void setKernel(Kernel kernel) {
		this.m_smo.setKernel(kernel);
	}
	
	/**
	 * Setting the C value for the Weka SMO classifier.
	 * @param c
	 */
	public void setC(double c) {
		this.m_smo.setC(c);
	}
	
	/**
	 * Getting the C value for the Weka SMO classifier.
	 * @return the C parameter
	 */
	public double getC() {
		return this.m_smo.getC();
	}
	
	
}
