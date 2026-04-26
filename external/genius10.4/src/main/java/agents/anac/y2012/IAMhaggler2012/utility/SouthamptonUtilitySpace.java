package agents.anac.y2012.IAMhaggler2012.utility;

import java.util.Enumeration;
import java.util.HashMap;

import genius.core.Bid;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

public class SouthamptonUtilitySpace {
	
	AdditiveUtilitySpace us;

	public SouthamptonUtilitySpace(AdditiveUtilitySpace us) {
		super();
		this.us = us;
	}

	/**
	 * @return a bid with the maximum utility value attainable in this util space
	 * @throws Exception
	 * @author C.R.Williams
	 */
	public final Bid getMaxUtilityBid() throws Exception
	{
    	HashMap<Integer, Value> bid = new HashMap<Integer, Value>();
        
        Objective root = us.getDomain().getObjectivesRoot();
        Enumeration<Objective> issueEnum = root.getPreorderIssueEnumeration();
        while(issueEnum.hasMoreElements()){
        	Objective is = issueEnum.nextElement();
    		bid.put(is.getNumber(), getMaxValue(is.getNumber()));
        }

		return new Bid(us.getDomain(), bid);
	}

    /**
     * @param pIssueIndex the index of the issue to get the maximum value for
     * @return a value that maximises the utility of the issue with the given index 
	 * @author C.R.Williams
     */
    private Value getMaxValue(int pIssueIndex)
    {
    	Evaluator lEvaluator = us.getEvaluator(pIssueIndex);
    	if(lEvaluator.getClass() == EvaluatorDiscrete.class)
    	{
    		return ((EvaluatorDiscrete)lEvaluator).getMaxValue();
    	}
    	if(lEvaluator.getClass() == EvaluatorInteger.class)
    	{
    		return getMaxValueInteger((EvaluatorInteger)lEvaluator);
    	}
    	if(lEvaluator.getClass() == EvaluatorReal.class)
    	{
    		return getMaxValueReal((EvaluatorReal)lEvaluator);
    	}
		return null;
	}
    
    /**
     * @return a value that maximises the utility
	 * @author C.R.Williams
     */
	private Value getMaxValueInteger(EvaluatorInteger eval) {
		double utility = 0;	
		switch(eval.getFuncType()) {
		case LINEAR:
			utility = maxLinear(eval.getSlope(), eval.getOffset());
			if (utility < eval.getLowerBound())
				utility = eval.getLowerBound();
			else if (utility > eval.getUpperBound())
				utility = eval.getUpperBound();
			break;
		default:
			return null;
		}
		return new ValueInteger((int)utility);
	}
	
    /**
     * @return a value that maximises the utility
	 * @author C.R.Williams
     */
	private Value getMaxValueReal(EvaluatorReal eval) {
		double utility = 0;	
		switch(eval.getFuncType()) {
		case LINEAR:
			utility = maxLinear(eval.getLinearParam(), eval.getConstantParam());
			if (utility < eval.getLowerBound())
				utility = eval.getLowerBound();
			else if (utility > eval.getUpperBound())
				utility = eval.getUpperBound();
			break;
		case CONSTANT:
			utility = eval.getLowerBound();
			break;
		case TRIANGULAR:
		case TRIANGULAR_VARIABLE_TOP:
			utility = maxTriangular(eval.getLinearParam(), eval.getConstantParam(), eval.getTopParam());
			break;
		default:
			return null;
		}
		return new ValueReal(utility);
	}
	
	private static double maxLinear(double coef1, double coef0) {
		if(coef1 > 0)
			return Double.MAX_VALUE;
		else
			return Double.MIN_VALUE;
	}
	
	private static double maxTriangular(double lowerBound, double upperBound, double top) {
		return top;
	}
}
