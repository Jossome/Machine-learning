package nn.core;

import java.util.*;

public class MultiLayerBackPropagationNeuralNetwork extends BackPropagationNeuralNetwork {

    /**
	 * Construct and return a new MultiLayerBackPropagationNeuralNetwork with the given
	 * InputUnits for inputs, NeuronUnits for outputs, int[] for hidden layer structure
	 * and double for learning rate (alpha). Length of hidden is the number of hidden
	 * layers and each element of hidden is the number of units in each hidden layer.
	 */
	public MultiLayerBackPropagationNeuralNetwork(InputUnit[] inputs, SigmoidNeuronUnit[] outputs, int[] hidden, double alpha, int epochs) {
		super(new Unit[2 + hidden.length][]);
		this.layers[0] = inputs;
		for (int i = 0; i < inputs.length; i++) this.layers[0][i] = new InputUnit();
		this.layers[1 + hidden.length] = outputs;

		// Constructing connections
		for (int i = 1; i <= hidden.length; i++) {
            this.layers[i] = new SigmoidNeuronUnit[hidden[i - 1]];
            for (int j = 0; j < hidden[i - 1]; j++) {
                this.layers[i][j] = new SigmoidNeuronUnit(this.layers[i - 1].length);
                for (int k = 0; k < this.layers[i - 1].length; k++) {
                    ((NeuronUnit) this.layers[i][j]).setInputUnit(k, this.layers[i - 1][k]);
                }
            }
		}
		for (int j = 0; j < outputs.length; j++) {
            this.layers[1 + hidden.length][j] = new SigmoidNeuronUnit(this.layers[hidden.length].length);
            for (int k = 0; k < this.layers[hidden.length].length; k++) {
                ((NeuronUnit) this.layers[1 + hidden.length][j]).setInputUnit(k, this.layers[hidden.length][k]);
            }
        }

		this.inputs = inputs;
		this.outputs = outputs;
		this.alpha = alpha;
		this.epochs = epochs;
	}

	protected double alpha;
	protected int epochs;
	protected InputUnit[] inputs;
	protected NeuronUnit[] outputs;

	public InputUnit[] getInputs() {
		return this.inputs;
	}

	public NeuronUnit[] getOutputs() {
		return this.outputs;
	}
	
	public void dump() {
		for (int i = 1; i < this.layers.length; i++) {
			for (int j = 0; j < this.layers[i].length; j++) {
				NeuronUnit unit = (NeuronUnit) this.layers[i][j];
				System.out.print(j);
				for (double w : unit.getWeights()) {
					System.out.format("\t%.2f", w);
				}
				System.out.println();
			}
			System.out.println("======");
		}
	}
	
	public void setRandomWeights() {
		for (int i = 1; i < this.layers.length; i++) {
            for (int j = 0; j < this.layers[i].length; j++) {
                for (int k = 0; k < this.layers[i - 1].length; k++) {
                    ((NeuronUnit) this.layers[i][j]).setWeight(k, (new Random()).nextDouble());
                }
            }
		}
	}
	
	public void backPropLearning(ArrayList<Example> examples) {
		int cnt = 0; // Used for counting epochs.
		do {
			this.setRandomWeights();
			for (Example eg: examples) {
				
				// Pass input value to the network.
				for (int i = 0; i < eg.inputs.length; i++) {
					// This line may affect 
					// this.layers[0][i] = new InputUnit();
					((InputUnit) this.layers[0][i]).setOutput(eg.inputs[i]);
				}
				
				// Propagate the inputs forward to compute the outputs
				for (int i = 1; i < this.layers.length; i++) {
					for (int j = 0; j < this.layers[i].length; j++) { 
						// for each node do these: 
						this.layers[i][j].fire();
					}
				}
				
				// Propagate deltas backward from output layer to input layer
				for (int i = 0; i < this.outputs.length; i++) {
					SigmoidNeuronUnit unit = (SigmoidNeuronUnit) this.outputs[i];
					double a_j = unit.getOutput();
					// g'(in) = a_j * (1 - a_j) for logistic activator functions
					unit.delta = a_j * (1 - a_j) * (eg.outputs[i] - a_j);
					
					// Update weight
					for(int k = 0; k < unit.numInputs(); k++) {
						unit.setWeight(k, unit.getWeight(k) + this.alpha * unit.delta * unit.getInputValue(k));
					}
				}
				
				for (int i = this.layers.length - 2; i >= 1; i--) {
					for (int j = 0; j < this.layers[i].length; j++) {
						SigmoidNeuronUnit unit = (SigmoidNeuronUnit) this.layers[i][j];
						double a_j = unit.getOutput();
						
						double sum = 0;
						for (int k = 0; k < this.layers[i + 1].length; k++) {
							SigmoidNeuronUnit tmp = (SigmoidNeuronUnit) this.layers[i + 1][k];
							sum += tmp.getWeight(j) * tmp.delta;
						}
											
						unit.delta = a_j * (1 - a_j) * sum;
						
						// Update weight
						for(int k = 0; k < unit.numInputs(); k++) {
							unit.setWeight(k, unit.getWeight(k) + this.alpha * unit.delta * unit.getInputValue(k));
						}
					}
				}
			}
			
		} while (++cnt < this.epochs);
		
	}
	
	static public void main(String[] args) {
		
		double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
		double[] Y = {0.1, 0.2};
		
		ArrayList<Example> examples = new ArrayList<Example>();
		examples.add(new Example(X, Y));
		examples.add(new Example(X, Y));
		examples.add(new Example(X, Y));
		
		InputUnit[] in = new InputUnit[X.length];
		SigmoidNeuronUnit[] out = new SigmoidNeuronUnit[Y.length];
		int[] x = {2,3};
		
		MultiLayerBackPropagationNeuralNetwork nn = new MultiLayerBackPropagationNeuralNetwork(in, out, x, 0.1, 10);
		nn.backPropLearning(examples);
		nn.dump();
	}
	

}
