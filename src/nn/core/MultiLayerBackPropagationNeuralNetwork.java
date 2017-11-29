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
	public MultiLayerBackPropagationNeuralNetwork(InputUnit[] inputs, SigmoidNeuronUnit[] outputs, int[] hidden, double alpha, int epochs, double threshold) {
		super(new Unit[2 + hidden.length][]);
		this.layers[0] = inputs;
		for (int i = 0; i < inputs.length; i++) this.layers[0][i] = new InputUnit();
		this.layers[1 + hidden.length] = outputs;

		// Constructing connections
        if (hidden.length > 0) {
            // If no hidden layer, which is a single layer nn.
    		for (int i = 1; i <= hidden.length; i++) {
                this.layers[i] = new SigmoidNeuronUnit[hidden[i - 1]];
                for (int j = 0; j < hidden[i - 1]; j++) {
                    this.layers[i][j] = new SigmoidNeuronUnit(this.layers[i - 1].length);
                    for (int k = 0; k < this.layers[i - 1].length; k++) {
                        ((NeuronUnit) this.layers[i][j]).setInputUnit(k, this.layers[i - 1][k]);
                    }
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
        this.threshold = threshold;
	}

	protected double alpha;
	protected int epochs;
    protected double threshold;
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
	
	public double accuracy(List<Example> examples) {
		int correct = 0;			
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
			
			for (int i = 0; i < this.outputs.length; i++) {
				SigmoidNeuronUnit unit = (SigmoidNeuronUnit) this.outputs[i];
				int a_j = unit.getOutput() >= 0.5 ? 1 : 0;
				if (a_j - eg.outputs[i] < 0.1) correct++;
			}
		}
		
		double acc = (double)correct / examples.size();
		// System.out.println(acc);
		return acc;
	}
	
	public void backPropLearning(ArrayList<Example> examples) {
		int cnt = 0; // Used for counting epochs.
		double error = 0.0;
		
		// Split dataset into 7:3 as training and test sets
		Collections.shuffle(examples);
		int split = (new Double(examples.size() * 0.7)).intValue();
		List<Example> train = examples.subList(0, split);
		List<Example> test = examples.subList(split, examples.size());
		
	    this.setRandomWeights();
		do {
			error = 0.0;
			for (Example eg: train) {
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
					error += unit.delta * unit.delta;
					
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
			error /= examples.size();
			
			this.trainingReport(test, cnt + 1, error);
			
		} while (++cnt < this.epochs && error > this.threshold);
		
	}

    public ArrayList<Example> readIrisExamples(String filename) {
		ArrayList<Example> examples = new ArrayList<Example>();
		
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			String line = "";
            while ((line = br.readLine()) != null) {
                String[] sample = line.split(",");
                double[] X = Arrays.stream(Arrays.copyOfRange(sample, 0, sample.length - 1)).mapToDouble(Double::parseDouble).toArray();
        		double[] Y = new double[1];
                switch (sample[sample.length - 1]) {
        		case "Iris-setosa":
        			Y[0] = 1; break;
        		case "Iris-versicolor":
        			Y[0] = 2; break;
        		case "Iris-virginica":
        			Y[0] = 3; break;
        		}
        		examples.add(new Example(X, Y));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return examples;
    }
    
	static public void main(String[] args) {
		
        // Filename, input dimension, hidden layers dimension, output dimension, epochs, 
        // alpha(learning rate), threshold(if error reaches below this, stop learning)
        String filename = args[0];
        int indim = Integer.parseInt(args[1]);
        int[] hidden = Arrays.stream(Arrays.copyOfRange(args, 2, args.length - 4)).mapToInt(Integer::parseInt).toArray();
        int outdim = Integer.parseInt(args[args.length - 4]);
        int epoch = Integer.parseInt(args[args.length - 3]);
        double alpha = Double.parseDouble(args[args.length - 2]);
        double threshold = Double.parseDouble(args[args.length - 1]);
	   
        System.out.println("NN structure: " + indim + "->" + Arrays.toString(hidden) + "->" + outdim);
        System.out.println("Epochs: " + epoch);
        System.out.println("Learning rate: " + alpha);
        System.out.println("Stop criteria: " + threshold);
        System.out.println("Start training:");
        
		InputUnit[] in = new InputUnit[indim];
		SigmoidNeuronUnit[] out = new SigmoidNeuronUnit[outdim];
	
		MultiLayerBackPropagationNeuralNetwork nn = new MultiLayerBackPropagationNeuralNetwork(in, out, hidden, alpha, epoch, threshold);
        ArrayList<Example> examples = nn.readIrisExamples(filename);
		nn.backPropLearning(examples);
		// nn.dump();
	}
	

}
