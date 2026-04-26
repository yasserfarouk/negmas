package agents.anac.y2017.tucagent;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EVALFUNCTYPE;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorReal;

public class BayesianOpponentModel extends OpponentModel {
	private AdditiveUtilitySpace fUS;
	private WeightHypothesis[] fWeightHyps;
	private ArrayList<ArrayList<EvaluatorHypothesis>> fEvaluatorHyps;
	private ArrayList<EvaluatorHypothesis[]> fEvalHyps;
	private ArrayList<UtilitySpaceHypothesis> fUSHyps;
	private boolean fUseMostProbableHypsOnly = false;
	private ArrayList<UtilitySpaceHypothesis> fMostProbableUSHyps;
	private double fPreviousBidUtility;
	private double EXPECTED_CONCESSION_STEP = 0.035D;
	private double SIGMA = 0.25D;
	private boolean USE_DOMAIN_KNOWLEDGE = false;
	List<Issue> issues;

	public BayesianOpponentModel(AdditiveUtilitySpace pUtilitySpace) {
		if (pUtilitySpace == null)
			throw new NullPointerException("pUtilitySpace=null");
		fDomain = pUtilitySpace.getDomain();
		fPreviousBidUtility = 1.0D;
		fUS = pUtilitySpace;
		fBiddingHistory = new ArrayList();
		issues = fDomain.getIssues();
		int lNumberOfHyps = factorial(issues.size());
		fWeightHyps = new WeightHypothesis[lNumberOfHyps];

		int index = 0;
		double[] P = new double[issues.size()];

		for (int i = 0; i < issues.size(); i++) {
			P[i] = ((i + 1) / (issues.size() * (fDomain.getIssues().size() + 1) / 2.0D));
		}
		antilex(new Integer(index), fWeightHyps, P, fDomain.getIssues().size() - 1);

		for (int i = 0; i < fWeightHyps.length; i++) {
			fWeightHyps[i].setProbability(1.0D / fWeightHyps.length);
		}

		fEvaluatorHyps = new ArrayList();
		int lTotalTriangularFns = 1;
		for (int i = 0; i < fUS.getNrOfEvaluators(); i++) {
			ArrayList<EvaluatorHypothesis> lEvalHyps = new ArrayList();
			lEvalHyps = new ArrayList();
			fEvaluatorHyps.add(lEvalHyps);
			switch (fUS.getEvaluator(((Issue) issues.get(i)).getNumber()).getType()) {

			case OBJECTIVE:
				IssueReal lIssue = (IssueReal) fDomain.getIssues().get(i);
				EvaluatorReal lHypEval = new EvaluatorReal();

				if (USE_DOMAIN_KNOWLEDGE) {
					lHypEval = new EvaluatorReal();
					lHypEval.setUpperBound(lIssue.getUpperBound());
					lHypEval.setLowerBound(lIssue.getLowerBound());
					lHypEval.setType(EVALFUNCTYPE.LINEAR);
					lHypEval.addParam(1, 1.0D / (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
					lHypEval.addParam(0,
							-lHypEval.getLowerBound() / (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
					EvaluatorHypothesis lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEval);
					lEvaluatorHypothesis.setDesc("uphill");
					lEvalHyps.add(lEvaluatorHypothesis);
				} else {
					lHypEval = new EvaluatorReal();
					lHypEval.setUpperBound(lIssue.getUpperBound());
					lHypEval.setLowerBound(lIssue.getLowerBound());
					lHypEval.setType(EVALFUNCTYPE.LINEAR);
					lHypEval.addParam(1, 1.0D / (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
					lHypEval.addParam(0,
							-lHypEval.getLowerBound() / (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
					EvaluatorHypothesis lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEval);
					lEvaluatorHypothesis.setDesc("uphill");
					lEvalHyps.add(lEvaluatorHypothesis);

					lHypEval = new EvaluatorReal();
					lHypEval.setUpperBound(lIssue.getUpperBound());
					lHypEval.setLowerBound(lIssue.getLowerBound());
					lHypEval.setType(EVALFUNCTYPE.LINEAR);
					lHypEval.addParam(1, -1.0D / (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
					lHypEval.addParam(0,
							1.0D + lHypEval.getLowerBound() / (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
					lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEval);
					lEvalHyps.add(lEvaluatorHypothesis);
					lEvaluatorHypothesis.setDesc("downhill");

					for (int k = 1; k <= lTotalTriangularFns; k++) {
						lHypEval = new EvaluatorReal();
						lHypEval.setUpperBound(lIssue.getUpperBound());
						lHypEval.setLowerBound(lIssue.getLowerBound());
						lHypEval.setType(EVALFUNCTYPE.TRIANGULAR);
						lHypEval.addParam(0, lHypEval.getLowerBound());
						lHypEval.addParam(1, lHypEval.getUpperBound());
						lHypEval.addParam(2, lHypEval.getLowerBound() + k
								* (lHypEval.getUpperBound() - lHypEval.getLowerBound()) / (lTotalTriangularFns + 1));
						lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEval);
						lEvaluatorHypothesis.setProbability(0.3333333333333333D);
						lEvalHyps.add(lEvaluatorHypothesis);
						lEvaluatorHypothesis.setDesc("triangular");
					}
				}
				for (int k = 0; k < lEvalHyps.size(); k++) {
					((EvaluatorHypothesis) lEvalHyps.get(k)).setProbability(1.0D / lEvalHyps.size());
				}

				break;

			case DISCRETE:
				lEvalHyps = new ArrayList();
				fEvaluatorHyps.add(lEvalHyps);

				IssueDiscrete lDiscIssue = (IssueDiscrete) fDomain.getIssues().get(i);
				if (USE_DOMAIN_KNOWLEDGE) {
					EvaluatorDiscrete lDiscreteEval = new EvaluatorDiscrete();
					for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++)
						lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), Integer.valueOf(1000 * j));
					EvaluatorHypothesis lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
					lEvaluatorHypothesis.setDesc("uphill");
					lEvalHyps.add(lEvaluatorHypothesis);
				} else {
					EvaluatorDiscrete lDiscreteEval = new EvaluatorDiscrete();
					for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++)
						lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), Integer.valueOf(1000 * j + 1));
					EvaluatorHypothesis lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
					lEvaluatorHypothesis.setDesc("uphill");
					lEvalHyps.add(lEvaluatorHypothesis);

					lDiscreteEval = new EvaluatorDiscrete();
					for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++) {
						lDiscreteEval.addEvaluation(lDiscIssue.getValue(j),
								Integer.valueOf(1000 * (lDiscIssue.getNumberOfValues() - j - 1) + 1));
					}
					lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
					lEvalHyps.add(lEvaluatorHypothesis);
					lEvaluatorHypothesis.setDesc("downhill");

					lDiscreteEval = new EvaluatorDiscrete();
					int halfway = lDiscIssue.getNumberOfValues() / 2;
					for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++) {
						if (j < halfway) {
							lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), Integer.valueOf(1000 * j + 1));
						} else {
							lDiscreteEval.addEvaluation(lDiscIssue.getValue(j),
									Integer.valueOf(1000 * (lDiscIssue.getNumberOfValues() - j - 1) + 1));
						}
					}
					lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
					lEvalHyps.add(lEvaluatorHypothesis);
					lEvaluatorHypothesis.setDesc("triangular");
				}

				break;
			}

		}

		buildEvaluationHyps();

		buildUniformHyps();
	}

	private void buildUniformHyps() {
		fUSHyps = new ArrayList();
		for (int i = 0; i < fWeightHyps.length; i++) {

			for (int j = 0; j < fEvalHyps.size(); j++) {
				UtilitySpaceHypothesis lUSHyp = new UtilitySpaceHypothesis(fDomain, fUS, fWeightHyps[i],
						(EvaluatorHypothesis[]) fEvalHyps.get(j));
				fUSHyps.add(lUSHyp);
			}
		}

		for (int i = 0; i < fUSHyps.size(); i++) {
			((UtilitySpaceHypothesis) fUSHyps.get(i)).setProbability(1.0D / fUSHyps.size());
		}
	}

	private void reverse(double[] P, int m) {
		int i = 0;
		int j = m;
		while (i < j) {
			double lTmp = P[i];
			P[i] = P[j];
			P[j] = lTmp;
			i++;
			j--;
		}
	}

	private Integer antilex(Integer index, WeightHypothesis[] hyps, double[] P, int m) {
		if (m == 0) {
			WeightHypothesis lWH = new WeightHypothesis(fDomain);
			for (int i = 0; i < P.length; i++)
				lWH.setWeight(i, P[i]);
			hyps[index.intValue()] = lWH;
			index = Integer.valueOf(index.intValue() + 1);
		} else {
			for (int i = 0; i <= m; i++) {
				index = antilex(index, hyps, P, m - 1);
				if (i < m) {
					double lTmp = P[i];
					P[i] = P[m];
					P[m] = lTmp;
					reverse(P, m - 1);
				}
			}
		}
		return index;
	}

	private double conditionalDistribution(double pUtility, double pPreviousBidUtility) {
		if (pPreviousBidUtility < pUtility) {
			return 0.0D;
		}

		double x = (pPreviousBidUtility - pUtility) / pPreviousBidUtility;
		double lResult = 1.0D / (SIGMA * Math.sqrt(6.283185307179586D)) * Math.exp(-(x * x) / (2.0D * SIGMA * SIGMA));
		return lResult;
	}

	public void updateBeliefs(Bid pBid) throws Exception {
		fBiddingHistory.add(pBid);
		if (haveSeenBefore(pBid)) {
			return;
		}
		double lFullProb = 0.0D;
		double lMaxProb = 0.0D;
		for (int i = 0; i < fUSHyps.size(); i++) {
			UtilitySpaceHypothesis hyp = (UtilitySpaceHypothesis) fUSHyps.get(i);
			double condDistrib = hyp.getProbability() * conditionalDistribution(
					((UtilitySpaceHypothesis) fUSHyps.get(i)).getUtility(pBid), fPreviousBidUtility);
			lFullProb += condDistrib;
			if (condDistrib > lMaxProb)
				lMaxProb = condDistrib;
			hyp.setProbability(condDistrib);
		}
		if (fUseMostProbableHypsOnly) {
			fMostProbableUSHyps = new ArrayList();
		}
		double lMostProbableHypFullProb = 0.0D;
		for (int i = 0; i < fUSHyps.size(); i++) {
			UtilitySpaceHypothesis hyp = (UtilitySpaceHypothesis) fUSHyps.get(i);
			double normalizedProbability = hyp.getProbability() / lFullProb;
			hyp.setProbability(normalizedProbability);
			if ((fUseMostProbableHypsOnly) && (normalizedProbability > lMaxProb * 0.99D / lFullProb)) {
				fMostProbableUSHyps.add(hyp);
				lMostProbableHypFullProb += normalizedProbability;
			}
		}
		if (fUseMostProbableHypsOnly) {
			for (int i = 0; i < fMostProbableUSHyps.size(); i++) {
				UtilitySpaceHypothesis hyp = (UtilitySpaceHypothesis) fMostProbableUSHyps.get(i);
				double normalizedProbability = hyp.getProbability() / lMostProbableHypFullProb;
				hyp.setProbability(normalizedProbability);
			}
		}

		System.out.println("BA: Using " + String.valueOf(fMostProbableUSHyps.size()) + " out of "
				+ String.valueOf(fUSHyps.size()) + "hyps");
		System.out.println(getMaxHyp().toString());

		fPreviousBidUtility -= EXPECTED_CONCESSION_STEP;
	}

	private void buildEvaluationHypsRecursive(ArrayList<EvaluatorHypothesis[]> pHyps, EvaluatorHypothesis[] pEval,
			int m) {
		if (m == 0) {
			ArrayList<EvaluatorHypothesis> lEvalHyps = (ArrayList) fEvaluatorHyps.get(fUS.getNrOfEvaluators() - 1);
			for (int i = 0; i < lEvalHyps.size(); i++) {
				pEval[(fUS.getNrOfEvaluators() - 1)] = ((EvaluatorHypothesis) lEvalHyps.get(i));
				EvaluatorHypothesis[] lTmp = new EvaluatorHypothesis[fUS.getNrOfEvaluators()];

				for (int j = 0; j < lTmp.length; j++)
					lTmp[j] = pEval[j];
				pHyps.add(lTmp);
			}
		} else {
			ArrayList<EvaluatorHypothesis> lEvalHyps = (ArrayList) fEvaluatorHyps.get(fUS.getNrOfEvaluators() - m - 1);
			for (int i = 0; i < lEvalHyps.size(); i++) {
				pEval[(fUS.getNrOfEvaluators() - m - 1)] = ((EvaluatorHypothesis) lEvalHyps.get(i));
				buildEvaluationHypsRecursive(pHyps, pEval, m - 1);
			}
		}
	}

	private void buildEvaluationHyps() {
		fEvalHyps = new ArrayList();
		EvaluatorHypothesis[] lTmp = new EvaluatorHypothesis[fUS.getNrOfEvaluators()];
		buildEvaluationHypsRecursive(fEvalHyps, lTmp, fUS.getNrOfEvaluators() - 1);
	}

	public double getExpectedUtility(Bid pBid) throws Exception {
		double lExpectedUtility = 0.0D;
		if ((fUseMostProbableHypsOnly) && (fMostProbableUSHyps != null)) {
			for (int i = 0; i < fMostProbableUSHyps.size(); i++) {
				UtilitySpaceHypothesis lUSHyp = (UtilitySpaceHypothesis) fMostProbableUSHyps.get(i);
				double p = lUSHyp.getProbability();
				double u = lUSHyp.getUtility(pBid);
				lExpectedUtility += p * u;
			}
		} else {
			for (int i = 0; i < fUSHyps.size(); i++) {
				UtilitySpaceHypothesis lUSHyp = (UtilitySpaceHypothesis) fUSHyps.get(i);
				double p = lUSHyp.getProbability();
				double u = lUSHyp.getUtility(pBid);
				lExpectedUtility += p * u;
			}
		}
		return lExpectedUtility;
	}

	public double getExpectedWeight(int pIssueNumber) {
		double lExpectedWeight = 0.0D;
		for (int i = 0; i < fUSHyps.size(); i++) {
			UtilitySpaceHypothesis lUSHyp = (UtilitySpaceHypothesis) fUSHyps.get(i);
			double p = lUSHyp.getProbability();
			double u = lUSHyp.getHeightHyp().getWeight(pIssueNumber);
			lExpectedWeight += p * u;
		}
		return lExpectedWeight;
	}

	public double getNormalizedWeight(Issue i, int startingNumber) {
		double sum = 0.0D;
		for (Issue issue : fDomain.getIssues()) {
			sum += getExpectedWeight(issue.getNumber() - startingNumber);
		}
		return getExpectedWeight(i.getNumber() - startingNumber) / sum;
	}

	private UtilitySpaceHypothesis getMaxHyp() {
		UtilitySpaceHypothesis lHyp = (UtilitySpaceHypothesis) fUSHyps.get(0);
		for (int i = 0; i < fUSHyps.size(); i++) {
			if (lHyp.getProbability() < ((UtilitySpaceHypothesis) fUSHyps.get(i)).getProbability())
				lHyp = (UtilitySpaceHypothesis) fUSHyps.get(i);
		}
		return lHyp;
	}

	private int factorial(int n) {
		if (n <= 1) {
			return 1;
		}
		return n * factorial(n - 1);
	}

	public void setMostProbableUSHypsOnly(boolean value) {
		fUseMostProbableHypsOnly = value;
	}
}
