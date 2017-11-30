package dt.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

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
		System.out.println(attributes);
		System.out.println(mostImportantVariable(attributes, examples));
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
		System.out.println(countOutputValue);
		int maxCount = 0;
		String commonValue = null;
		for (String s: countOutputValue.keySet()) {
			if (countOutputValue.get(s) > maxCount) {
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
		if (countOutputValue.size() == 1) {
			String uniqueValue = countOutputValue.keySet().iterator().next();
			return uniqueValue;
		}
		else {
			//cannot return null, since the type of return is String
			return "null";
		}
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
	
}
