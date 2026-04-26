package genius.core.uncertainty;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

/**
 * Service class for building an additive utility space of the form:
 * u(x_1, ..., x_n) = w_1 * e_1(x_1) + ... + w_n * e_n(x_n)
 */
public class AdditiveUtilitySpaceFactory 
{
	private AdditiveUtilitySpace u;
	
	/**
	 * Generates an simple Utility Space on the domain, with equal weights and zero values.
	 * Everything is zero-filled to already have all keys contained in the utility maps.
	 */
	public AdditiveUtilitySpaceFactory(Domain d) 
	{
		List<Issue> issues = d.getIssues();
		int noIssues = issues.size();
		Map<Objective, Evaluator> evaluatorMap = new HashMap<Objective, Evaluator>();
		for (Issue i : issues) {
			IssueDiscrete issue = (IssueDiscrete) i;
			EvaluatorDiscrete evaluator = new EvaluatorDiscrete();
			evaluator.setWeight(1.0 / noIssues);
			for (ValueDiscrete value : issue.getValues()) {
				evaluator.setEvaluationDouble(value, 0.0);
			}
			evaluatorMap.put(issue, evaluator);
		}
				
		u = new AdditiveUtilitySpace(d, evaluatorMap);
	}
	
	/**
	 * Sets w_i := weight 
	 */
	public void setWeight(Issue i, double weight)
	{
		EvaluatorDiscrete evaluator = (EvaluatorDiscrete) u.getEvaluator(i);
		evaluator.setWeight(weight);
	}
	
	/**
	 * Sets e_i(v) := value 
	 */
	public void setUtility(Issue i, ValueDiscrete v, double value)
	{
		EvaluatorDiscrete evaluator = (EvaluatorDiscrete) u.getEvaluator(i);
		if (evaluator == null)
		{
			evaluator = new EvaluatorDiscrete();
			u.addEvaluator(i, evaluator);
		}
		evaluator.setEvaluationDouble(v, value);
	}
	
	public double getUtility(Issue i, ValueDiscrete v)
	{
		EvaluatorDiscrete evaluator = (EvaluatorDiscrete) u.getEvaluator(i);
		return evaluator.getDoubleValue(v);
	}
	
	/**
	 * A simple heuristic for estimating a discrete {@link AdditiveUtilitySpace} from a {@link BidRanking}.
	 * Gives 0 points to all values occurring in the lowest ranked bid, 
	 * then 1 point to all values occurring in the second lowest bid, and so on.
	 */
	public void estimateUsingBidRanks(BidRanking r)
	{
		// Retrieve the issues of the domain only once
		List<Issue> issues = r.getBidIssues();
		double points = 0;
		
		for (Bid b : r.getBidOrder())
		{
			for (Issue i : issues)
			{
				int no = i.getNumber();
				ValueDiscrete v = (ValueDiscrete) b.getValue(no);
				double oldUtil = getUtility(i, v);
				setUtility(i, v, oldUtil + points);
			}
			points += 1;
		}
		normalizeWeightsByMaxValues();
	}
	
	private void normalizeWeightsByMaxValues()
	{
		for (Issue i : getIssues())
		{
			EvaluatorDiscrete evaluator = (EvaluatorDiscrete) u.getEvaluator(i);
			evaluator.normalizeAll();
		}
		scaleAllValuesFrom0To1();
		u.normalizeWeights();
	}
	
	public void scaleAllValuesFrom0To1()
	{
		for (Issue i : getIssues())
		{
			EvaluatorDiscrete evaluator = (EvaluatorDiscrete) u.getEvaluator(i);
			evaluator.scaleAllValuesFrom0To1();
		}
	}
	
	public void normalizeWeights()
	{
		u.normalizeWeights();
	}
	
	/**
	 * Returns the utility space that has been created.
	 */
	public AdditiveUtilitySpace getUtilitySpace() 
	{
		return u;
	}

	/**
	 * In this class, we can assume all issues are of the type {@link IssueDiscrete}.
	 * @return all issues in the domain.
	 */
	public List<IssueDiscrete> getIssues() 
	{
		List<IssueDiscrete> issues = new ArrayList<>();
		for (Issue i : getDomain().getIssues()) 
		{
			IssueDiscrete issue = (IssueDiscrete) i;
			issues.add(issue);
		}
		return issues;
	}
	
	public Domain getDomain() {
		return u.getDomain();
	}

}
