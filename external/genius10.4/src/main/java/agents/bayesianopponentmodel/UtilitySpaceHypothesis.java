package agents.bayesianopponentmodel;

import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.utility.AdditiveUtilitySpace;

public class UtilitySpaceHypothesis extends Hypothesis {
	private WeightHypothesis fWeightHyp;
	private EvaluatorHypothesis[] fEvalHyp;
	private Domain fDomain;
	private AdditiveUtilitySpace fUS;
	List<Issue> issues;

	public UtilitySpaceHypothesis(Domain pDomain, AdditiveUtilitySpace pUS,
			WeightHypothesis pWeightHyp, EvaluatorHypothesis[] pEvalHyp) {
		fUS = pUS;
		fDomain = pDomain;
		issues = fDomain.getIssues();
		fWeightHyp = pWeightHyp;
		fEvalHyp = pEvalHyp;
	}

	public Domain getDomain() {
		return fDomain;
	}

	public AdditiveUtilitySpace getUtilitySpace() {
		return fUS;
	}

	public EvaluatorHypothesis[] getEvalHyp() {
		return fEvalHyp;
	}

	public WeightHypothesis getHeightHyp() {
		return fWeightHyp;
	}

	public double getUtility(Bid pBid) {
		double u = 0;

		for (int k = 0; k < fEvalHyp.length; k++) {
			try {
				u = u
						+ fWeightHyp.getWeight(k)
						* fEvalHyp[k].getEvaluator().getEvaluation(fUS, pBid,
								issues.get(k).getNumber());
			} catch (Exception e) {
				u = 0;
			}
		}
		return u;

	}

	public String toString() {
		String lResult = "";
		lResult += fWeightHyp.toString();
		for (EvaluatorHypothesis lHyp : fEvalHyp) {
			lResult += lHyp.toString() + ";";
		}
		lResult += String.format("%1.5f", getProbability());
		return lResult;
	}
}
