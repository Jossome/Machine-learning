package lc.core;

import java.util.List;
import java.util.Random;

import math.util.VectorOps;
import lc.core.*;

public class PerceptronClassifier extends LinearClassifier {
	
	public PerceptronClassifier(double[] weights) {
		super(weights);
	}
	
	public PerceptronClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * Update the weights of this LinearClassifer using the given
	 * inputs/output example and learning rate alpha.
	 */
	public void update(double[] x, double y, double alpha) {
		for (int i = 0; i < x.length; i++) {
			this.weights[i] = this.weights[i] + alpha * (y - this.eval(x)) * x[i];
		}
		
	}
	
	/**
	 * Threshold the given value using this LinearClassifier's
	 * threshold function.
	 */
	public double threshold(double z) {
		System.out.println(z);
		if (z >= 0) {
			return 1.0;
		}
		else {
			return 0.0;
		}
	}
	
	
	
}
