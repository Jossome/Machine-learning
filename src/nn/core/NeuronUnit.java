package nn.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for the non-input units of a NeuralNetwork.
 */
abstract public class NeuronUnit implements Unit {
	
	public NeuronUnit(int ninputs) {
		this.inputs = new Unit[ninputs];
		this.weights = new double[ninputs];
		this.inputs[0] = new ConstantUnit(1.0);
	}
	
	/**
	 * Vector of Units connected to this NeuronUnit's inputs.
	 * The first element of this vector, a_0, is automatically
	 * initialized to a ConstantUnit with value 1.0 (AIMA p. 728).
	 */
	protected Unit[] inputs;
	
	public int numInputs() {
		return this.inputs.length;
	}
	
	/**
	 * List of NeuronUnits whose inputs are connected to this
	 * NeuronUnit's output. This is used for backprop.
	 * Kind of ugly to use a list for this, but who cares once
	 * the network is built.
	 */
	public List<NeuronUnit> outgoingConnections = new ArrayList<NeuronUnit>();
	
	/**
	 * Connect the given unit as this NeuronUnit's i'th input.
	 * If the unit is itself a NeuronUnit, then we also add
	 * this NeuornUnit to its list of outgoing connections.
	 */
	public void setInputUnit(int i, Unit u) {
		this.inputs[i] = u;
		if (u instanceof NeuronUnit) {
			NeuronUnit nu = (NeuronUnit)u;
			nu.outgoingConnections.add(this);
		}
	}
	
	/**
	 * The vector of weights for this NeuronUnit's inputs.
	 */
	protected double[] weights;
	
	public void setWeight(int i, double w) {
		this.weights[i] = w;
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public double getWeight(int i) {
		return this.weights[i];
	}
	
	/**
	 * This NeuronUnit's output value.
	 */
	protected double output = 0.0;
	
	/**
	 * Return the output value of this Unit.
	 */
	@Override
	public double getOutput() {
		return output;
	}

	/**
	 * Return the weighted sum of this NeuronUnit's inputs.
	 * This is used in the backpropagation algorithm.
	 */
	public double getInputSum() {
		double sum = 0.0;
		for (int i=0; i < inputs.length; i++) {
			sum +=  this.weights[i] * this.inputs[i].getOutput();
		}
		return sum;
	}
	
	/**
	 * ``Each unit j first computes a weighted sum of its inputs.
	 * Then it applies an activation function g to this sum to
	 * derive the output.'' (AIMA p728).
	 */
	@Override
	public void fire() {
		this.output = this.activation(this.getInputSum());
	}
	
	/**
	 * The activation function used by this NeuronUnit.
	 * This method must be specified by subclassses.
	 */
	abstract public double activation(double in);
	
	/**
	 * Update the weights of this NeuronUnit using the given
	 * input values, the given output value, and learning rate (alpha).
	 */
	@Override
	abstract public void update(double[] inputs, double output, double alpha);

	/**
	 * Error term computed during backprop.
	 */
	public double delta;
	
}
