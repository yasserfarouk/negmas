package agents.anac.y2017.mamenchis;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.UtilitySpace;
import genius.core.xml.SimpleElement;

import java.util.Set;

public class PreferModel {

	private UtilitySpace utilitySpace;
	private Set<Entry<Objective, Evaluator>> evaluators;
	private int numOfIssues;
	private double[] weights;
	private int[] indexOfMaxFreq;
	private int numOfValues;
	private int learnRate;

	public PreferModel(UtilitySpace utilitySpace, int num) {
		this.utilitySpace = utilitySpace;
		numOfIssues = num;
		weights = new double[num];
		initWeights();
		initEvaluators();
		setNumOfValues();
		learnRate = 1;
	}

	public Set<Entry<Objective, Evaluator>> getEvaluators() {
		return evaluators;
	}

	private void setEvaluators(Set<Entry<Objective, Evaluator>> evaluators) {
		this.evaluators = evaluators;
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double weights[]) {
		this.weights = weights;
	}

	public int[] getIndexOfMaxFreq() {
		return indexOfMaxFreq;
	}

	public void setIndexOfMaxFreq(int[] indexOfMaxFreq) {
		this.indexOfMaxFreq = indexOfMaxFreq;
	}

	public int getNumOfValues() {
		return numOfValues;
	}

	public void setNumOfValues() {
		int num = 0;
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		for (Issue issue : issues) {
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete discrete = (IssueDiscrete) issue;
				num += discrete.getNumberOfValues();
				break;
			case INTEGER:
				IssueInteger integer = (IssueInteger) issue;
				int distance = integer.getUpperBound() - integer.getLowerBound();
				num += (distance + 1);
				break;
			default:
				break;
			}
		}
		numOfValues = num;
	}

	private void initEvaluators() {
		try {
			Object[] issuesXML = ((SimpleElement) (utilitySpace.toXML().getChildByTagName("objective")[0]))
					.getChildByTagName("issue").clone();
			Map<Objective, Evaluator> map = new LinkedHashMap<>();
			for (int i = 0; i < numOfIssues; i++) {
				Issue issue = utilitySpace.getDomain().getIssues().get(i);
				switch (issue.getType()) {
				case DISCRETE:
					List<ValueDiscrete> values = ((IssueDiscrete) utilitySpace.getDomain().getIssues().get(i))
							.getValues();
					EvaluatorDiscrete ed = new EvaluatorDiscrete();
					ed.loadFromXML((SimpleElement) issuesXML[i]);
					ed.setWeight(weights[i]);
					for (ValueDiscrete item : values) {
						ed.addEvaluation(item, (int) 1);
					}
					map.put(issue, ed);
					break;
				case INTEGER:
					EvaluatorInteger et = new EvaluatorInteger();
					et.loadFromXML((SimpleElement) issuesXML[i]);
					et.setWeight(weights[i]);
					et.setLinearFunction(0.0, 1.0);
					map.put(issue, et);
					break;
				default:
					System.err.println("Unsuported issue type " + issue.getType());
					break;
				}
				setEvaluators((map.entrySet()));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void loadEvaluatorFromMyPrefer() {
		try {
			Object[] issuesXML = ((SimpleElement) (utilitySpace.toXML().getChildByTagName("objective")[0]))
					.getChildByTagName("issue").clone();
			Object[] weightXML = ((SimpleElement) (utilitySpace.toXML().getChildByTagName("objective")[0]))
					.getChildByTagName("weight");
			Map<Objective, Evaluator> map = new LinkedHashMap<>();
			for (int i = 0; i < numOfIssues; i++) {
				Issue issue = utilitySpace.getDomain().getIssues().get(i);
				switch (issue.getType()) {
				case DISCRETE:
					EvaluatorDiscrete ed = new EvaluatorDiscrete();
					ed.loadFromXML((SimpleElement) issuesXML[i]);
					ed.setWeight(Double.parseDouble(((SimpleElement) weightXML[i]).getAttribute("value")));
					map.put(issue, ed);
					break;
				case INTEGER:
					EvaluatorInteger et = new EvaluatorInteger();
					et.loadFromXML((SimpleElement) issuesXML[i]);
					et.setWeight(Double.parseDouble(((SimpleElement) weightXML[i]).getAttribute("value")));
					map.put(issue, et);
					break;
				default:
					System.err.println("Unsuported issue type " + issue.getType());
					break;
				}
				setEvaluators((map.entrySet()));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void updateEvaluators(Bid bid, int learnRate) {
		HashMap<Integer, Value> values = bid.getValues();
		List<Issue> issues = bid.getIssues();
		int i = 0;
		for (Entry<Objective, Evaluator> evaluator : evaluators) {
			evaluator.getValue().setWeight(weights[i]);
			try {
				Value value = values.get(i + 1);
				Issue issue = issues.get(i);
				for (int j = 0; j < numOfIssues; j++) {
					if (evaluator.getKey().toString().equals(issues.get(j).toString())) {
						value = values.get(j + 1);
						issue = issues.get(j);
						break;
					}
				}
				switch (value.getType()) {
				case DISCRETE:
					ValueDiscrete valueDiscrete = (ValueDiscrete) values.get(i + 1);
					int old = ((EvaluatorDiscrete) evaluator.getValue()).getEvaluationNotNormalized(valueDiscrete);
					((EvaluatorDiscrete) evaluator.getValue()).setEvaluation(value, old + learnRate);
					break;
				case INTEGER:
					EvaluatorInteger ei = ((EvaluatorInteger) evaluator.getValue());
					if (ei.weightLocked()) {
						continue;
					}
					IssueInteger issueInteger = (IssueInteger) issue;
					ValueInteger valueInteger = (ValueInteger) value;
					int iValue = valueInteger.getValue();
					int distanceToUpper = Math.abs(issueInteger.getUpperBound() - iValue);
					int distanceToLower = Math.abs(issueInteger.getLowerBound() - iValue);
					if (distanceToUpper < distanceToLower) {
						ei.setLinearFunction(0.0, 1.0);
					} else {
						ei.setLinearFunction(1.0, 0.0);
					}
					ei.lockWeight();
					break;
				default:
					break;
				}

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			i++;
		}

	}

	private void initWeights() {
		for (int i = 0; i < weights.length; i++) {
			weights[i] = 1D / (double) numOfIssues;
		}
	}

	public void updateEvaluators(Bid bid) {
		boolean hasUpadated = false;
		HashMap<Integer, Value> values = bid.getValues();
		List<Issue> issues = bid.getIssues();
		int i = 0;
		for (Entry<Objective, Evaluator> evaluator : evaluators) {
			try {
				Value value = values.get(i + 1);
				Issue issue = issues.get(i);
				for (int j = 0; j < numOfIssues; j++) {
					if (evaluator.getKey().toString().equals(issues.get(j).toString())) {
						value = values.get(j + 1);
						issue = issues.get(j);
						break;
					}
				}
				switch (value.getType()) {
				case DISCRETE:
					ValueDiscrete valueDiscrete = (ValueDiscrete) values.get(i + 1);
					int old = ((EvaluatorDiscrete) evaluator.getValue()).getEvaluationNotNormalized(valueDiscrete);
					((EvaluatorDiscrete) evaluator.getValue()).setEvaluation(value, old + 1);
					hasUpadated = true;
					break;
				case INTEGER:
					EvaluatorInteger ei = ((EvaluatorInteger) evaluator.getValue());
					if (ei.weightLocked()) {
						continue;
					}
					IssueInteger issueInteger = (IssueInteger) issue;
					ValueInteger valueInteger = (ValueInteger) value;
					int iValue = valueInteger.getValue();
					int distanceToUpper = Math.abs(issueInteger.getUpperBound() - iValue);
					int distanceToLower = Math.abs(issueInteger.getLowerBound() - iValue);
					if (distanceToUpper < distanceToLower) {
						ei.setLinearFunction(0.0, 1.0);
					} else {
						ei.setLinearFunction(1.0, 0.0);
					}
					hasUpadated = true;
					ei.lockWeight();
					break;
				default:
					break;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			i++;
		}
		if (hasUpadated) {
			learnRate--;
		}
	}

	public double getUtil(Bid bid) {
		double util = 0D;
		Map<Integer, Value> values = bid.getValues();
		int issueNr = 1;
		for (Entry<Objective, Evaluator> evaluator : evaluators) {
			try {
				Value value = values.get(issueNr);
				double weight = evaluator.getValue().getWeight();
				double valueEvaluation = 0.0;
				switch (value.getType()) {
				case DISCRETE:
					EvaluatorDiscrete dEvaluator = (EvaluatorDiscrete) evaluator.getValue();
					valueEvaluation = dEvaluator.getEvaluation((ValueDiscrete) value);
					util += weight * valueEvaluation;
					break;
				case INTEGER:
					EvaluatorInteger iEvaluator = (EvaluatorInteger) evaluator.getValue();
					valueEvaluation = iEvaluator.getEvaluation(Integer.parseInt(value.toString()));
					util += weight * valueEvaluation;
					break;
				default:
					break;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			issueNr++;
		}
		return util;
	}

	public double getValueEvaluation(int issueNr, Value value) {
		double evaluation = 0.0;
		int i = 1;
		for (Entry<Objective, Evaluator> evaluator : evaluators) {
			if (i != issueNr) {
				i++;
				continue;
			} else {
				switch (value.getType()) {
				case DISCRETE:
					ValueDiscrete dValue = (ValueDiscrete) value;
					EvaluatorDiscrete dEvaluator = (EvaluatorDiscrete) evaluator.getValue();
					try {
						evaluation = dEvaluator.getWeight() * dEvaluator.getEvaluation(dValue);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					break;
				case INTEGER:
					ValueInteger vInteger = (ValueInteger) value;
					EvaluatorInteger iEvaluator = (EvaluatorInteger) evaluator.getValue();
					evaluation = iEvaluator.getWeight() * iEvaluator.getEvaluation(vInteger.getValue());
					break;
				default:
					System.err.println("Unsupported value type: " + value.getType());
					break;
				}
				i++;
			}
		}
		return evaluation;
	}
}
