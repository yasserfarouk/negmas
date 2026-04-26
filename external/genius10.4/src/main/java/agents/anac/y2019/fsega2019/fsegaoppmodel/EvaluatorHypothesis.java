package agents.anac.y2019.fsega2019.fsegaoppmodel;

import genius.core.utility.Evaluator;

public class EvaluatorHypothesis extends Hypothesis
{
	private Evaluator dEval;
	private String sDescription;
	
	public EvaluatorHypothesis (Evaluator pEval, String pDescription)
	{
		dEval = pEval;
		sDescription = pDescription;
	}
	
	public Evaluator getEvaluator()
	{
		return dEval;
	}

	public String getDescription()
	{
		return sDescription;
	}
	
	public String toString()
	{
		return "Evaluator hypothesis " + sDescription + ": " + dEval.toString();
	}
}

