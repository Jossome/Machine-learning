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
        int k = Integer.parseInt(args[1]);
		for (Example e: examples) {
			System.out.println(e);
		}
		DecisionTreeLearner learner = new DecisionTreeLearner(problem);
		DecisionTree tree = learner.learn(examples);
		tree.dump();
		tree.test(examples);
		double error_rate = learner.crossValidation(learner, examples, k);
		System.out.format("cross validation error rate: %f%%, correct: %f%%\n", error_rate, 100 - error_rate);
	}

}
