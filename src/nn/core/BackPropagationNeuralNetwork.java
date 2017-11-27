package nn.core;

public class BackPropagationNeuralNetwork extends NeuralNetwork {

	protected Unit[][] layers;

	public BackPropagationNeuralNetwork(Unit[][] layers) {
		this.layers = layers;
	}

}
