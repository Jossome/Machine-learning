package nn.core;

import java.util.List;
import java.util.Random;
import java.util.Set;

import nn.core.Example;

/**
 * A SingleLayerFeedForwardNeuralNetwork is a single-layer feed-forward network
 * where all the inputs are directly connected to the outputs.
 * (AIMA Section 18.7.2)
 */
public class SingleLayerFeedForwardNeuralNetwork extends FeedForwardNeuralNetwork {
	
	/**
	 * Construct and return a new SingleLayerFeedForwardNeuralNetwork with the given
	 * InputUnits for inputs and NeuronUnits for outputs. It's up
	 * to you to arrange the feed-forward connections between the Units
	 * properly.
	 */
	public SingleLayerFeedForwardNeuralNetwork(InputUnit[] inputs, NeuronUnit[] outputs) {
		super(new Unit[2][]);
		this.layers[0] = inputs;
		this.layers[1] = outputs;
		this.inputs = inputs;
		this.outputs = outputs;
	}
	
	protected InputUnit[] inputs;
	protected NeuronUnit[] outputs;
	
	public InputUnit[] getInputs() {
		return this.inputs;
	}

	public NeuronUnit[] getOutputs() {
		return this.outputs;
	}
	
	/**
	 * Print this SingleLayerFeedForwardNeuralNetwork to stdout.
	 * All we need to print are the weights on the output units. 
	 */
	public void dump() {
		for (int i=0; i < this.outputs.length; i++) {
			NeuronUnit unit = this.outputs[i];
			System.out.print(i);
			for (double w : unit.getWeights()) {
				System.out.format("\t%.2f", w);
			}
			System.out.println();
		}
	}
	
	/**
	 * Train this SingleLayerFeedForwardNeuralNetwork on the given Examples,
	 * using given learning rate alpha.
	 * This means updating the weights on the output units for
	 * each example on each step.
	 */
	public void train(List<Example> examples, double alpha) {
		for (int i=0; i < examples.size(); i++) {
			Example ex = examples.get(i);
			for (int j=0; j < this.outputs.length; j++) {
				this.outputs[j].update(ex.inputs, ex.outputs[j], alpha);
			}
			this.trainingReport(examples, i);
		}
	}

}
