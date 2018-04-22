package nn.core;

import java.util.List;

public abstract class BackPropagationNeuralNetwork extends NeuralNetwork {

	protected Unit[][] layers;

	public BackPropagationNeuralNetwork(Unit[][] layers) {
		this.layers = layers;
	}

	@Override
	abstract double accuracy(List<Example> examples);

}
