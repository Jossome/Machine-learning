package dt.example;

import java.io.File;
import java.io.IOException;
import java.util.Set;

import dt.core.DecisionTree;
import dt.core.DecisionTreeLearner;
import dt.core.Domain;
import dt.core.Example;
import dt.core.Problem;
import dt.core.Variable;

public class IrisDataProblem extends Problem{

	public IrisDataProblem() {
		super();
		// Input variables
		Domain sizeDomain = new Domain("S", "MS", "L", "ML");
		this.inputs.add(new Variable("sepalLength", sizeDomain));
		this.inputs.add(new Variable("sepalWidth", sizeDomain));
		this.inputs.add(new Variable("petalLength", sizeDomain));
		this.inputs.add(new Variable("petalWidth", sizeDomain));
		// Output variable
		this.output = new Variable("irisClass", new Domain("Iris-setosa", "Iris-versicolor", "Iris-virginica"));
	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		IrisDataProblem problem = new IrisDataProblem();
		problem.dump();
		Set<Example> examples = problem.readExamplesFromCSVFile(new File(args[0]));
		for (Example e: examples) {
			System.out.println(e);
		}
		DecisionTree tree = new DecisionTreeLearner(problem).learn(examples);
		tree.dump();
		tree.test(examples);
	}

}
