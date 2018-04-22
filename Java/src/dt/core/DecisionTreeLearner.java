package dt.core;

import java.util.ArrayList;
//import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Collections;

//import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils.Collections;

import dt.util.ArraySet;

public class DecisionTreeLearner extends AbstractDecisionTreeLearner{

	public DecisionTreeLearner(Problem problem) {
		super(problem);
	}
	
	/**
	 * Main recursive decision-tree learning (ID3) method.
	 * This must be implemented by subclasses.  
	 */
	public DecisionTree learn(Set<Example> examples, List<Variable> attributes, Set<Example> parent_examples) {
		
		if (examples.isEmpty()) {
			return new DecisionTree(pluralityValue(parent_examples));
		}
		else if (!uniqueOutputValue(examples).equals("null")) {
			return new DecisionTree(uniqueOutputValue(examples));
		}
		else if (attributes.isEmpty()) {
			return new DecisionTree(pluralityValue(examples));
		}
		else {
			DecisionTree dTree;
			Variable A = mostImportantVariable(attributes, examples);
			dTree = new DecisionTree(A);
			for (String vk: A.domain) {
				Set<Example> exs = examplesWithValueForAttribute(examples, A, vk);
				
				// deep copy attributes
				List<Variable> attributesMinusA = new ArrayList<Variable>();
				for (Variable var: attributes) {
					if (!var.equals(A)) {
						attributesMinusA.add(var);
					}
				}
				DecisionTree subtree = learn(exs, attributesMinusA, examples);
				dTree.children.add(subtree);
			}
			return dTree;
		}
		
	}
	
	/**
	 * Returns the most common output value among a set of Examples,
	 * breaking ties randomly (or not--it may not matter to you).
	 */
	public String pluralityValue(Set<Example> examples) {
		HashMap<String, Integer> countOutputValue = new HashMap<String, Integer>();
		for (Example e: examples) {
			String outputValue = e.getOutputValue();
			if (!countOutputValue.containsKey(outputValue)) {
				countOutputValue.put(outputValue, 1);
			}
			else {
				countOutputValue.put(outputValue, countOutputValue.get(outputValue) + 1);
			}
		}
		int maxCount = 0;
		String commonValue = null;
		for (String s: countOutputValue.keySet()) {
			if (countOutputValue.get(s) >= maxCount) {
				maxCount = countOutputValue.get(s);
				commonValue = s;
			}
		}
		return commonValue;
	}
	
	/**
	 * Returns the single unique output value among the given examples
	 * is there is only one, otherwise null.
	 */
	public String uniqueOutputValue(Set<Example> examples) {
		Iterator<Example> iter = examples.iterator();
		String unique = iter.next().getOutputValue();
		while(iter.hasNext()) {
			if(!unique.equals(iter.next().getOutputValue())) {
				return "null";
			}
		}
		return unique;
	}
	
	/**
	 * Return the subset of the given examples for which Variable a has value vk.
	 */
	public Set<Example> examplesWithValueForAttribute(Set<Example> examples, Variable a, String vk) {
		Set<Example> vkExamples = new ArraySet<Example>();
		for (Example e: examples) {
			if (e.getInputValue(a).equals(vk)) {
				vkExamples.add(e);
			}
		}
		return vkExamples;
	}
	
	/**
	 * Return the number of the given examples for which Variable a has value vk.
	 */
	public int countExamplesWithValueForAttribute(Set<Example> examples, Variable a, String vk) {
		return examplesWithValueForAttribute(examples, a, vk).size();
	}
	
	/**
	 * Return the number of the given examples for which the output has value vk.
	 */
	public int countExamplesWithValueForOutput(Set<Example> examples, String vk) {
		int count = 0;
		for (Example e: examples) {
			if (e.getOutputValue().equals(vk)) {
				count += 1;
			}
		}
		return count;
	}
	
	public double crossValidation(DecisionTreeLearner learner, Set<Example> examples, int k) {
		double error = 0;
		
		ArrayList<Example> list_examples = new ArrayList<Example>(examples); 
		Collections.shuffle(list_examples);
		
		for (int fold = 1; fold <= k; fold++) {
			int split1 = (new Double(list_examples.size() * (fold-1) / k)).intValue();
			int split2 = (new Double(list_examples.size() * fold / k - 1)).intValue();
			Set<Example> train = new ArraySet<Example>(list_examples.subList(0, split1));
			train.addAll(list_examples.subList(split2, list_examples.size()));
			Set<Example> test = new ArraySet<Example>(list_examples.subList(split1, split2));
			DecisionTree tree = learner.learn(train);
			error += tree.error_rate(test);
		}

		return error / k;
	}
	
}
