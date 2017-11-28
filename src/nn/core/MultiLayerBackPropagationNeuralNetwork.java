package nn.core;

import java.io.*;
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
		ArrayList<Example> examples = new ArrayList<Example>();
		
		try (BufferedReader br = new BufferedReader(new FileReader("D:/Jossome/Programs/AI/project4/Machine-learning/src/nn/core/iris.data.txt"))) {
			String line = "";
            while ((line = br.readLine()) != null) {
                String[] sample = line.split(",");
                double[] X = Arrays.stream(Arrays.copyOfRange(sample, 0, sample.length - 1)).mapToDouble(Double::parseDouble).toArray();
        		double[] Y = new double[1];
                switch (sample[sample.length - 1]) {
        		case "Iris-setosa":
        			Y[0] = 1;
        		case "Iris-versicolor":
        			Y[0] = 2;
        		case "Iris-virginica":
        			Y[0] = 3;
        		}
        		examples.add(new Example(X, Y));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
		
		InputUnit[] in = new InputUnit[4];
		SigmoidNeuronUnit[] out = new SigmoidNeuronUnit[1];
		int[] x = {3,2};
		
		MultiLayerBackPropagationNeuralNetwork nn = new MultiLayerBackPropagationNeuralNetwork(in, out, x, 0.1, 10);
		nn.dump();
		nn.backPropLearning(examples);
		nn.dump();
	}
	

}
