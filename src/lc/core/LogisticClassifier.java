package lc.core;

import java.util.List;
import java.util.Random;

import math.util.VectorOps;
import lc.core.*;
import java.lang.Math;

public class LogisticClassifier extends LinearClassifier{
	
	public LogisticClassifier(double[] weights) {
		super(weights);
	}
	
	public LogisticClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * Update the weights of this LinearClassifer using the given
	 * inputs/output example and learning rate alpha.
	 */
	public void update(double[] x, double y, double alpha) {
		for (int i = 0; i < x.length; i++) {
			double h = this.eval(x);
			this.weights[i] = this.weights[i] + alpha * (y - h) * h * (1.0 - h) * x[i];
		}
		
	}
	
	/**
	 * Threshold the given value using this LinearClassifier's
	 * threshold function.
	 */
	public double threshold(double z) {
		double e = Math.E;
		return 1.0 / (1.0 + Math.pow(e, -z));
	}

}
