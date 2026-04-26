package agents.anac.y2010.AgentFSEGA;

import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.utility.AdditiveUtilitySpace;

public class UtilitySpaceHypothesis extends Hypothesis {
	private Domain dDomain;
	private AdditiveUtilitySpace dUS;
	private WeightHypothesis dWeightHyp;
	private EvaluatorHypothesis[] eEvalHyp;
	List<Issue> issues;

	public UtilitySpaceHypothesis(Domain pDomain, AdditiveUtilitySpace pUS,
			WeightHypothesis pWeightHyp, EvaluatorHypothesis[] pEvalHyp) {
		dUS = pUS;
		dDomain = pDomain;
		issues = dDomain.getIssues();
		dWeightHyp = pWeightHyp;
		eEvalHyp = pEvalHyp;
	}

	public Domain getDomain() {
		return dDomain;
	}

	public EvaluatorHypothesis[] getEvalHyp() {
		return eEvalHyp;
	}

	public WeightHypothesis getHeightHyp() {
		return dWeightHyp;
	}

	public double getUtility(Bid pBid) {
		double u = 0;

		for (int k = 0; k < eEvalHyp.length; k++) {
			try {
				// utility = sum( weight * isue_evaluation)
				u = u
						+ dWeightHyp.getWeight(k)
						* eEvalHyp[k].getEvaluator().getEvaluation(dUS, pBid,
								issues.get(k).getNumber());
			} catch (Exception e) {
				u = 0;
			}
		}
		return u;

	}

	public String toString() {
		String lResult = "UtilitySpaceHypotesis[";
		lResult += dWeightHyp.toString() + ", EvaluatorHypotesis[";
		for (EvaluatorHypothesis lHyp : eEvalHyp) {
			lResult += lHyp.toString() + "; ";
		}
		lResult += String.format("], %1.5f]", getProbability());
		return lResult;
	}
}
