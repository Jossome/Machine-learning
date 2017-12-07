package lc.example;

import java.io.IOException;
import java.util.List;

import lc.core.Example;
import lc.core.LearningRateSchedule;
import lc.core.LogisticClassifier;
import lc.display.ClassifierDisplay;

public class LogisticClassifierTest {

	/**
	 * Train a PerceptronClassifier on a file of examples and
	 * print its accuracy after each training step.
	 */
	public static void main(String[] argv) throws IOException {
		if (argv.length < 3) {
			System.out.println("usage: java LogisticClassifierTest data-filename nsteps alpha");
			System.out.println("       specify alpha=0 to use decaying learning rate schedule (AIMA p725)");
			System.exit(-1);
		}
		String filename = argv[0];
		int nsteps = Integer.parseInt(argv[1]);
		double alpha = Double.parseDouble(argv[2]);
		System.out.println("filename: " + filename);
		System.out.println("nsteps: " + nsteps);
		System.out.println("alpha: " + alpha);
		
		ClassifierDisplay display = new ClassifierDisplay("LogisticClassifier: " + filename);
		List<Example> examples = Data.readFromFile(filename);
		int ninputs = examples.get(0).inputs.length; 
		LogisticClassifier classifier = new LogisticClassifier(ninputs) {
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
			
			public void trainingReport(List<Example> examples, int stepnum, int nsteps) {
				double accuracy = accuracy(examples);
				System.out.println(stepnum + "\t" + accuracy);
				display.addPoint(stepnum/(double)nsteps, accuracy);
			}
		};
		if (alpha > 0) {
			classifier.train(examples, nsteps, alpha);
			
		} else {
			classifier.train(examples, 100000, new LearningRateSchedule() {
				public double alpha(int t) { return 1000.0/(1000.0+t); }
			});
		}
		
		//cross validation
		double error_rate = classifier.crossValidation(examples, 10, nsteps, alpha);
		System.out.format("cross validation error rate: %f%%, correct: %f%%\n", error_rate * 100, (1 - error_rate) * 100);
	}

}

