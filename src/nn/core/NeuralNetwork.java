package nn.core;

import java.util.List;

import nn.core.Example;

/**
 * Base class for NeuralNetworks made up of Units.
 */
abstract public class NeuralNetwork {

	/**
	 * This method is called after each weight update during training.
	 * Subclasses can override it to gather statistics or update displays.
	 */
	protected void trainingReport(List<Example> examples, int stepnum, double error) {
		System.out.println("Epoch: " + stepnum + "\tAccuracy: " + accuracy(examples) + "\tMSE: " + error);
	}
	
	abstract double accuracy(List<Example> examples);
}
