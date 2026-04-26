package agents.bayesianopponentmodel;

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

/**
 * Implementation of the scalable Bayesian Model.
 * 
 * Opponent Modelling in Automated Multi-Issue Negotiation Using Bayesian
 * Learning by K. Hindriks, D. Tykhonov
 * 
 * KNOWN BUGS: 1. Opponent model does not take the opponent's strategy into
 * account, in contrast to the original paper which depicts an assumption about
 * the opponent'strategy which adapts over time.
 * 
 * 2. The opponent model becomes invalid after a while as NaN occurs in some
 * hypotheses, corrupting the overall estimation.
 */
public class BayesianOpponentModelScalable extends OpponentModel {

	private AdditiveUtilitySpace fUS;
	private ArrayList<ArrayList<WeightHypothesis2>> fWeightHyps;
	private ArrayList<ArrayList<EvaluatorHypothesis>> fEvaluatorHyps;
	// private ArrayList<EvaluatorHypothesis[]> fEvalHyps;
	// public ArrayList<Bid> fBiddingHistory; // previous bids of the opponent,
	// not our bids.
	// private ArrayList<UtilitySpaceHypothesis> fUSHyps;
	private double fPreviousBidUtility;
	List<Issue> issues;
	private double[] fExpectedWeights;

	public BayesianOpponentModelScalable(AdditiveUtilitySpace pUtilitySpace) {
		//

		fPreviousBidUtility = 1;
		fDomain = pUtilitySpace.getDomain();
		issues = fDomain.getIssues();
		fUS = pUtilitySpace;
		fBiddingHistory = new ArrayList<Bid>();
		fExpectedWeights = new double[pUtilitySpace.getDomain().getIssues()
				.size()];
		fWeightHyps = new ArrayList<ArrayList<WeightHypothesis2>>();
		// generate all possible ordering combinations of the weights

		initWeightHyps();
		// generate all possible hyps of evaluation functions
		fEvaluatorHyps = new ArrayList<ArrayList<EvaluatorHypothesis>>();
		int lTotalTriangularFns = 4;
		for (int i = 0; i < fUS.getNrOfEvaluators(); i++) {
			ArrayList<EvaluatorHypothesis> lEvalHyps = new ArrayList<EvaluatorHypothesis>();

			switch (fUS.getEvaluator(issues.get(i).getNumber()).getType()) {

			case REAL:
				lEvalHyps = new ArrayList<EvaluatorHypothesis>();
				fEvaluatorHyps.add(lEvalHyps);
				// EvaluatorReal lEval = (EvaluatorReal)(fUS.getEvaluator(i));
				IssueReal lIssue = (IssueReal) (fDomain.getIssues().get(i));
				// uphill
				EvaluatorReal lHypEval = new EvaluatorReal();
				lHypEval.setUpperBound(lIssue.getUpperBound());
				lHypEval.setLowerBound(lIssue.getLowerBound());
				lHypEval.setType(EVALFUNCTYPE.LINEAR);
				lHypEval.addParam(1, (double) 1
						/ (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
				lHypEval.addParam(
						0,
						-lHypEval.getLowerBound()
								/ (lHypEval.getUpperBound() - lHypEval
										.getLowerBound()));
				EvaluatorHypothesis lEvaluatorHypothesis = new EvaluatorHypothesis(
						lHypEval);
				lEvaluatorHypothesis.setDesc("uphill");
				lEvalHyps.add(lEvaluatorHypothesis);
				// downhill
				lHypEval = new EvaluatorReal();
				lHypEval.setUpperBound(lIssue.getUpperBound());
				lHypEval.setLowerBound(lIssue.getLowerBound());
				lHypEval.setType(EVALFUNCTYPE.LINEAR);
				lHypEval.addParam(1, -(double) 1
						/ (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
				lHypEval.addParam(0, (double) 1 + lHypEval.getLowerBound()
						/ (lHypEval.getUpperBound() - lHypEval.getLowerBound()));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEval);
				lEvalHyps.add(lEvaluatorHypothesis);
				lEvaluatorHypothesis.setDesc("downhill");
				for (int k = 1; k <= lTotalTriangularFns; k++) {
					// triangular
					lHypEval = new EvaluatorReal();
					lHypEval.setUpperBound(lIssue.getUpperBound());
					lHypEval.setLowerBound(lIssue.getLowerBound());
					lHypEval.setType(EVALFUNCTYPE.TRIANGULAR);
					lHypEval.addParam(0, lHypEval.getLowerBound());
					lHypEval.addParam(1, lHypEval.getUpperBound());
					double lMaxPoint = lHypEval.getLowerBound()
							+ (double) k
							* (lHypEval.getUpperBound() - lHypEval
									.getLowerBound())
							/ (lTotalTriangularFns + 1);
					lHypEval.addParam(2, lMaxPoint);
					lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEval);
					lEvalHyps.add(lEvaluatorHypothesis);
					lEvaluatorHypothesis.setDesc("triangular "
							+ String.format("%1.2f", lMaxPoint));
				}
				for (int k = 0; k < lEvalHyps.size(); k++) {
					lEvalHyps.get(k).setProbability(
							(double) 1 / lEvalHyps.size());
				}

				break;
			case DISCRETE:
				lEvalHyps = new ArrayList<EvaluatorHypothesis>();
				fEvaluatorHyps.add(lEvalHyps);
				// EvaluatorReal lEval = (EvaluatorReal)(fUS.getEvaluator(i));
				IssueDiscrete lDiscIssue = (IssueDiscrete) (fDomain.getIssues()
						.get(i));
				// uphill
				EvaluatorDiscrete lDiscreteEval = new EvaluatorDiscrete();
				for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++)
					lDiscreteEval.addEvaluation(lDiscIssue.getValue(j),
							1000 * j + 1);
				lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
				lEvaluatorHypothesis.setProbability((double) 1 / 3);
				lEvaluatorHypothesis.setDesc("uphill");
				lEvalHyps.add(lEvaluatorHypothesis);
				// downhill
				lDiscreteEval = new EvaluatorDiscrete();
				for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++)
					lDiscreteEval
							.addEvaluation(
									lDiscIssue.getValue(j),
									1000 * (lDiscIssue.getNumberOfValues() - j - 1) + 1);
				lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
				lEvaluatorHypothesis.setProbability((double) 1 / 3);
				lEvalHyps.add(lEvaluatorHypothesis);
				lEvaluatorHypothesis.setDesc("downhill");
				if (lDiscIssue.getNumberOfValues() > 2) {
					lTotalTriangularFns = lDiscIssue.getNumberOfValues() - 1;
					for (int k = 1; k < lTotalTriangularFns; k++) {
						// triangular. Wouter: we need to CHECK this.
						lDiscreteEval = new EvaluatorDiscrete();
						for (int j = 0; j < lDiscIssue.getNumberOfValues(); j++)
							if (j < k) {
								lDiscreteEval.addEvaluation(
										lDiscIssue.getValue(j), 1000 * j / k);
							} else {
								// lEval =
								// (1.0-(double)(j-k)/(lDiscIssue.getNumberOfValues()-1.0-k));
								lDiscreteEval.addEvaluation(
										lDiscIssue.getValue(j),
										1000
												* (lDiscIssue
														.getNumberOfValues()
														- j - 1)
												/ (lDiscIssue
														.getNumberOfValues()
														- k - 1) + 1);
							}
						lEvaluatorHypothesis = new EvaluatorHypothesis(
								lDiscreteEval);
						lEvalHyps.add(lEvaluatorHypothesis);
						lEvaluatorHypothesis.setDesc("triangular "
								+ String.valueOf(k));
					}// for
				}// if
				for (int k = 0; k < lEvalHyps.size(); k++) {
					lEvalHyps.get(k).setProbability(
							(double) 1 / lEvalHyps.size());
				}
				break;
			}// switch
		}
		for (int i = 0; i < fExpectedWeights.length; i++)
			fExpectedWeights[i] = getExpectedWeight(i);

		// printEvalsDistribution();
	}

	void initWeightHyps() {
		int lWeightHypsNumber = 11;
		for (int i = 0; i < fUS.getDomain().getIssues().size(); i++) {
			ArrayList<WeightHypothesis2> lWeightHyps = new ArrayList<WeightHypothesis2>();
			for (int j = 0; j < lWeightHypsNumber; j++) {
				WeightHypothesis2 lHyp = new WeightHypothesis2(fDomain);
				lHyp.setProbability((1.0 - ((double) j + 1.0)
						/ lWeightHypsNumber)
						* (1.0 - ((double) j + 1.0) / lWeightHypsNumber)
						* (1.0 - ((double) j + 1.0) / lWeightHypsNumber));
				lHyp.setWeight((double) j / (lWeightHypsNumber - 1));
				lWeightHyps.add(lHyp);
			}
			double lN = 0;
			for (int j = 0; j < lWeightHypsNumber; j++) {
				lN += lWeightHyps.get(j).getProbability();
			}
			for (int j = 0; j < lWeightHypsNumber; j++) {
				lWeightHyps.get(j).setProbability(
						lWeightHyps.get(j).getProbability() / lN);
			}

			fWeightHyps.add(lWeightHyps);
		}
	}

	private double conditionalDistribution(double pUtility,
			double pPreviousBidUtility) {
		// TODO: check this condition
		// if(pPreviousBidUtility<pUtility) return 0;
		// else {
		double lSigma = 0.25;
		double x = (pPreviousBidUtility - pUtility) / pPreviousBidUtility;
		double lResult = 1.0 / (lSigma * Math.sqrt(2.0 * Math.PI))
				* Math.exp(-(x * x) / (2.0 * lSigma * lSigma));
		return lResult;
		// }
	}

	public double getExpectedEvaluationValue(Bid pBid, int pIssueNumber)
			throws Exception {
		double lExpectedEval = 0;
		for (int j = 0; j < fEvaluatorHyps.get(pIssueNumber).size(); j++) {
			lExpectedEval = lExpectedEval
					+ fEvaluatorHyps.get(pIssueNumber).get(j).getProbability()
					* fEvaluatorHyps
							.get(pIssueNumber)
							.get(j)
							.getEvaluator()
							.getEvaluation(fUS, pBid,
									issues.get(pIssueNumber).getNumber());
		}
		return lExpectedEval;

	}

	public double getExpectedWeight(int pIssueNumber) {
		double lExpectedWeight = 0;
		for (int i = 0; i < fWeightHyps.get(pIssueNumber).size(); i++) {
			lExpectedWeight += fWeightHyps.get(pIssueNumber).get(i)
					.getProbability()
					* fWeightHyps.get(pIssueNumber).get(i).getWeight();
		}
		return lExpectedWeight;
	}

	private double getPartialUtility(Bid pBid, int pIssueIndex)
			throws Exception {
		// calculate partial utility w/o issue pIssueIndex
		double u = 0;
		for (int j = 0; j < fDomain.getIssues().size(); j++) {
			if (pIssueIndex == j)
				continue;
			// calculate expected weight of the issue
			double w = 0;
			for (int k = 0; k < fWeightHyps.get(j).size(); k++)
				w += fWeightHyps.get(j).get(k).getProbability()
						* fWeightHyps.get(j).get(k).getWeight();
			u = u + w * getExpectedEvaluationValue(pBid, j);
		}
		return u;
	}

	public void updateWeights() throws Exception {
		Bid lBid = fBiddingHistory.get(fBiddingHistory.size() - 1);
		ArrayList<ArrayList<WeightHypothesis2>> lWeightHyps = new ArrayList<ArrayList<WeightHypothesis2>>();
		// make new hyps array
		for (int i = 0; i < fWeightHyps.size(); i++) {
			ArrayList<WeightHypothesis2> lTmp = new ArrayList<WeightHypothesis2>();
			for (int j = 0; j < fWeightHyps.get(i).size(); j++) {
				WeightHypothesis2 lHyp = new WeightHypothesis2(fUS.getDomain());
				lHyp.setWeight(fWeightHyps.get(i).get(j).getWeight());
				lHyp.setProbability(fWeightHyps.get(i).get(j).getProbability());
				lTmp.add(lHyp);
			}
			lWeightHyps.add(lTmp);
		}

		// for(int k=0;k<5;k++) {
		for (int j = 0; j < fDomain.getIssues().size(); j++) {
			double lN = 0;
			double lUtility = 0;
			for (int i = 0; i < fWeightHyps.get(j).size(); i++) {
				// if(!lBid.getValue(j).equals(lPreviousBid.getValue(j))) {
				lUtility = fWeightHyps.get(j).get(i).getWeight()
						* getExpectedEvaluationValue(lBid, j)
						+ getPartialUtility(lBid, j);
				lN += fWeightHyps.get(j).get(i).getProbability()
						* conditionalDistribution(lUtility, fPreviousBidUtility);
				/*
				 * } else { lN += fWeightHyps.get(j).get(i).getProbability(); }
				 */
			}
			// 2. receiveMessage probabilities
			for (int i = 0; i < fWeightHyps.get(j).size(); i++) {
				// if(!lBid.getValue(j).equals(lPreviousBid.getValue(j))) {
				lUtility = fWeightHyps.get(j).get(i).getWeight()
						* getExpectedEvaluationValue(lBid, j)
						+ getPartialUtility(lBid, j);
				lWeightHyps
						.get(j)
						.get(i)
						.setProbability(
								fWeightHyps.get(j).get(i).getProbability()
										* conditionalDistribution(lUtility,
												fPreviousBidUtility) / lN);
				/*
				 * } else {
				 * lWeightHyps.get(j).get(i).setProbability(fWeightHyps.
				 * get(j).get(i).getProbability()/lN); }
				 */
			}
		}
		// }
		fWeightHyps = lWeightHyps;
	}

	public void updateEvaluationFns() throws Exception {
		Bid lBid = fBiddingHistory.get(fBiddingHistory.size() - 1);
		// make new hyps array
		// for(int k=0;k<5;k++){
		ArrayList<ArrayList<EvaluatorHypothesis>> lEvaluatorHyps = new ArrayList<ArrayList<EvaluatorHypothesis>>();
		for (int i = 0; i < fEvaluatorHyps.size(); i++) {
			ArrayList<EvaluatorHypothesis> lTmp = new ArrayList<EvaluatorHypothesis>();
			for (int j = 0; j < fEvaluatorHyps.get(i).size(); j++) {
				EvaluatorHypothesis lHyp = new EvaluatorHypothesis(
						fEvaluatorHyps.get(i).get(j).getEvaluator());
				lHyp.setDesc(fEvaluatorHyps.get(i).get(j).getDesc());
				lHyp.setProbability(fEvaluatorHyps.get(i).get(j)
						.getProbability());
				lTmp.add(lHyp);
			}
			lEvaluatorHyps.add(lTmp);
		}

		// 1. calculate the normalization factor

		for (int i = 0; i < fDomain.getIssues().size(); i++) {
			// 1. calculate the normalization factor
			double lN = 0;
			for (int j = 0; j < fEvaluatorHyps.get(i).size(); j++) {
				EvaluatorHypothesis lHyp = fEvaluatorHyps.get(i).get(j);
				lN += lHyp.getProbability()
						* conditionalDistribution(
								getPartialUtility(lBid, i)
										+ getExpectedWeight(i)
										* (lHyp.getEvaluator().getEvaluation(
												fUS, lBid, issues.get(i)
														.getNumber())),
								fPreviousBidUtility);
			}
			// 2. receiveMessage probabilities
			for (int j = 0; j < fEvaluatorHyps.get(i).size(); j++) {
				EvaluatorHypothesis lHyp = fEvaluatorHyps.get(i).get(j);
				lEvaluatorHyps
						.get(i)
						.get(j)
						.setProbability(
								lHyp.getProbability()
										* conditionalDistribution(
												getPartialUtility(lBid, i)
														+ getExpectedWeight(i)
														* (lHyp.getEvaluator()
																.getEvaluation(
																		fUS,
																		lBid,
																		issues.get(
																				i)
																				.getNumber())),
												fPreviousBidUtility) / lN);
			}
		}
		fEvaluatorHyps = lEvaluatorHyps;
		// }
		printEvalsDistribution();
	}

	public boolean haveSeenBefore(Bid pBid) {
		for (Bid tmpBid : fBiddingHistory) {
			if (pBid.equals(tmpBid))
				return true;
		}
		return false;
	}

	public void updateBeliefs(Bid pBid) throws Exception {
		if (!isCrashed()) {
			if (haveSeenBefore(pBid))
				return;
			fBiddingHistory.add(pBid);

			// do not receiveMessage the bids if it is the first bid
			if (fBiddingHistory.size() > 1) {

				// receiveMessage the weights
				updateWeights();
				// receiveMessage evaluation functions
				updateEvaluationFns();
			} else {
				// do not receiveMessage the weights
				// receiveMessage evaluation functions
				updateEvaluationFns();
			} // if

			// System.out.println(getMaxHyp().toString());
			// calculate utility of the next partner's bid according to the
			// concession functions
			fPreviousBidUtility = fPreviousBidUtility - 0.003;
			for (int i = 0; i < fExpectedWeights.length; i++) {
				fExpectedWeights[i] = getExpectedWeight(i);
			}
			findMinMaxUtility();
			// printBestHyp();
		}
	}

	/**
	 * Plan: cache the results for pBid in a Hash table. empty the hash table
	 * whenever updateWeights or updateEvaluationFns is called.
	 * 
	 * @param pBid
	 * @return weeighted utility where weights represent likelihood of each
	 *         hypothesis
	 * @throws Exception
	 */
	public double getExpectedUtility(Bid pBid) throws Exception {
		// calculate expected utility
		double u = 0;
		for (int j = 0; j < fDomain.getIssues().size(); j++) {
			// calculate expected weight of the issue
			double w = fExpectedWeights[j];
			/*
			 * for(int k=0;k<fWeightHyps.get(j).size();k++) w +=
			 * fWeightHyps.get(
			 * j).get(k).getProbability()*fWeightHyps.get(j).get(
			 * k).getWeight();(
			 */
			u = u + w * getExpectedEvaluationValue(pBid, j);
		}

		return u;
	}

	private void printBestHyp() {
		double[] lBestWeights = new double[fUS.getDomain().getIssues().size()];
		EvaluatorHypothesis[] lBestEvals = new EvaluatorHypothesis[fUS
				.getDomain().getIssues().size()];
		for (int i = 0; i < fUS.getDomain().getIssues().size(); i++) {
			// find best weight
			double lMaxWeightProb = -1;
			for (int j = 0; j < fWeightHyps.get(i).size(); j++) {
				if (fWeightHyps.get(i).get(j).getProbability() > lMaxWeightProb) {
					lMaxWeightProb = fWeightHyps.get(i).get(j).getProbability();
					lBestWeights[i] = fWeightHyps.get(i).get(j).getWeight();
				}
			}
			// find best evaluation fn
			double lMaxEvalProb = -1;
			for (int j = 0; j < fEvaluatorHyps.get(i).size(); j++) {
				if (fEvaluatorHyps.get(i).get(j).getProbability() > lMaxEvalProb) {
					lMaxEvalProb = fEvaluatorHyps.get(i).get(j)
							.getProbability();
					lBestEvals[i] = fEvaluatorHyps.get(i).get(j);
				}
			}

		}
		/*
		 * //print all weights for(int
		 * i=0;i<fUS.getDomain().getIssues().size();i++) {
		 * System.out.print(String.format("%1.2f", getExpectedWeight(i))+";"); }
		 * //print all Evaluators for(int
		 * i=0;i<fUS.getDomain().getIssues().size();i++) {
		 * System.out.print(lBestEvals[i].getDesc()+";"); }
		 * System.out.println();
		 */
	}

	void printEvalsDistribution() {
		/*
		 * for(int i=0;i<fUS.getDomain().getIssues().size();i++) { for(int
		 * j=0;j<fEvaluatorHyps.get(i).size();j++)
		 * System.out.print(String.format("%1.2f",
		 * fEvaluatorHyps.get(i).get(j).getProbability())+";");
		 * System.out.println(); }
		 */

	}

	public double getNormalizedWeight(Issue i, int startingNumber) {
		double sum = 0;
		for (Issue issue : fDomain.getIssues()) {
			sum += getExpectedWeight(issue.getNumber() - startingNumber);
		}
		return (getExpectedWeight(i.getNumber() - startingNumber)) / sum;
	}

}
