package nn.core;

/**
 * Base class for feed-forward NeuralNetworks, where the connections
 * are only in one direction (that is, it forms a directed acyclic
 * graph). ``Feed-forward networks are arranged in layers, such that
 * each unit receives input only from units in the immediately preceding
 * layer.'' (AIMA p729).
 */
public class FeedForwardNeuralNetwork extends NeuralNetwork {
	
	/**
	 * The Units of this FeedForwardNeuralNetwork arranged in
	 * layers.
	 */
	protected Unit[][] layers;
	
	public FeedForwardNeuralNetwork(Unit[][] layers) {
		this.layers = layers;
	}

}
