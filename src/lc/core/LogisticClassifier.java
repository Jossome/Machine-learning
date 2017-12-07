package lc.core;

import java.util.ArrayList;
import java.util.Collections;
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
		double h = this.eval(x);
		for (int i = 0; i < x.length; i++) {
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
	
	public double crossValidation(List<Example> examples, int k, int nsteps, double alpha) {
		double error = 0;

		Collections.shuffle(examples);

		for (int fold = 1; fold <= k; fold++) {
			int split1 = (new Double(examples.size() * (fold - 1) / k)).intValue();
			int split2 = (new Double(examples.size() * fold / k)).intValue();
			List<Example> train = new ArrayList<Example>(examples.subList(0, split1));
			train.addAll(examples.subList(split2, examples.size()));
			List<Example> test = new ArrayList<Example>(examples.subList(split1, split2));
			LogisticClassifier classifier = new LogisticClassifier(train.get(0).inputs.length) {
				public double accuracy(List<Example> examples) {
					int ncorrect = 0;
					for (Example ex : examples) {
						double result = eval(ex.inputs);
						if (Math.abs(result - ex.output) <= 0.5) {
							ncorrect += 1;
						}
					}
					return (double)ncorrect / examples.size();
				}
			};
			if (alpha > 0) {
				classifier.train(examples, nsteps, alpha);
				
			} else {
				classifier.train(examples, 100000, new LearningRateSchedule() {
					public double alpha(int t) { return 1000.0/(1000.0+t); }
				});
			}
			error += classifier.get_last_accuracy(test);
		}

		return 1 - error / k;
	}

}
