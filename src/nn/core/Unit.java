package nn.core;

/**
 * Interface implemented by computational elements
 * (``units'') in a NeuralNetwork.
 */
public interface Unit {
	
	/**
	 * Return the output value of this Unit.
	 */
	public double getOutput();
	
	/**
	 * ``Fire'' this Unit by recomputing its output
	 * value given the values of its inputs.
	 */
	public void fire();
	
	/**
	 * Update this Unit's weights based on the given input values,
	 * output value, and learning rate.
	 */
	public void update(double[] inputs, double output, double alpha);


}
