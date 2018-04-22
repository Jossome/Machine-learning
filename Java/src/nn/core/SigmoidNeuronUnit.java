package nn.core;

public class SigmoidNeuronUnit extends NeuronUnit {

	public SigmoidNeuronUnit(int ninputs) {
		super(ninputs);
	}
    /**
	 * Use sigmoid function as activation function.
	 */
    @Override
    public double activation(double in) {
        return 1 / (1 + Math.exp(-in));
    }
    
	@Override
	public void update(double[] inputs, double output, double alpha) {
		// TODO Auto-generated method stub
		
	}
}
