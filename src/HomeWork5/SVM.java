package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
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
•	recurrence-events is the 0.0 class and will be the NEGATIVE class
•	no-recurrence-events is the 1.0 class and will be the POSITIVE class
	 * @param instances
	 * @return int array of size 4 in this order [TP, FP, TN, FN].
	 */
	public int[] calcConfusion(Instances instances) throws Exception{
		return null;
	}
	
	/**
	 * Setting the Weka SMO classifier kernel
	 * @param kernel
	 */
	public void setKernel(Kernel kernel) {
		
	}
	
	/**
	 * Setting the C value for the Weka SMO classifier.
	 * @param c
	 */
	public void setC(double c) {
		
	}
	
	/**
	 * Getting the C value for the Weka SMO classifier.
	 * @return the C parameter
	 */
	public double getC() {
		return 0;
	}
	
	
}
