package lc.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import math.util.VectorOps;
import lc.core.*;

public class PerceptronClassifier extends LinearClassifier {

	// public PerceptronClassifier(double[] weights) {
	// super(weights);
	// }

	public PerceptronClassifier(int ninputs) {
		super(ninputs);
	}

	/**
	 * Update the weights of this LinearClassifer using the given inputs/output
	 * example and learning rate alpha.
	 */
	public void update(double[] x, double y, double alpha) {
		double hwx = this.eval(x);
		for (int i = 0; i < x.length; i++) {
			this.weights[i] = this.weights[i] + alpha * (y - hwx) * x[i];
		}

	}

	/**
	 * Threshold the given value using this LinearClassifier's threshold function.
	 */
	public double threshold(double z) {
		if (z >= 0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}

	public double crossValidation(List<Example> examples, int k, int nsteps, double alpha) {
		double error = 0;

		Collections.shuffle(examples);

		for (int fold = 1; fold <= k; fold++) {
			int split1 = (new Double(examples.size() * (fold - 1) / k)).intValue();
			int split2 = (new Double(examples.size() * fold / k - 1)).intValue();
			List<Example> train = new ArrayList<Example>(examples.subList(0, split1));
			train.addAll(examples.subList(split2, examples.size()));
			List<Example> test = new ArrayList<Example>(examples.subList(split1, split2));
			PerceptronClassifier classifier = new PerceptronClassifier(train.get(0).inputs.length);
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
