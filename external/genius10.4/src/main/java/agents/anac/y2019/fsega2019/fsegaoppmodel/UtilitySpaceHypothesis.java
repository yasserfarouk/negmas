package agents.anac.y2019.fsega2019.fsegaoppmodel;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;



public class UtilitySpaceHypothesis extends Hypothesis
{
	private Domain dDomain;
	private UtilitySpace dUS;
	private WeightHypothesis dWeightHyp;
	private EvaluatorHypothesis[] eEvalHyp;
	ArrayList<Issue> issues;	

	public UtilitySpaceHypothesis(Domain pDomain, UtilitySpace pUS, WeightHypothesis pWeightHyp, EvaluatorHypothesis[] pEvalHyp)
	{
		dUS = pUS;
		dDomain = pDomain;
		issues =  (ArrayList<Issue>) dDomain.getIssues();
		dWeightHyp = pWeightHyp;
		eEvalHyp = pEvalHyp;		
	}

	public Domain getDomain()
	{
		return dDomain;
	}

	public EvaluatorHypothesis[] getEvalHyp()
	{
		return eEvalHyp;
	}

	public WeightHypothesis getHeightHyp()
	{
		return dWeightHyp;
	}
	
	public double getUtility(Bid pBid)
	{
		double u = 0;
		 
		for(int k = 0; k < eEvalHyp.length; k++)
		{
			try
			{
				//utility = sum( weight * isue_evaluation)
				u = u + dWeightHyp.getWeight(k) * eEvalHyp[k].getEvaluator().getEvaluation((AdditiveUtilitySpace) dUS, pBid, issues.get(k).getNumber());
			}
			catch (Exception e)
			{
				//TODO: for test
				//System.out.println("Exception: in FSEGAOpponentModel.getUtil: " + e.getMessage() + " using 0");
				
				u = 0;
			}
		}
		return u;
		
	}

    @Override
	public String toString()
	{
		String lResult = String.format("UtilitySpaceHypotesis      probab: %1.5f\n", getProbability());
		lResult += dWeightHyp.toString() + "  - EvaluatorHypotesis:\n";
		for(EvaluatorHypothesis lHyp : eEvalHyp)
		{
            lResult += "        - ";
			lResult += lHyp.toString() + ";\n";
		}

        lResult += "\n\n";
		return lResult;
	}
}
