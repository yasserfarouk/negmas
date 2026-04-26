package agents.anac.y2014.DoNA;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AbstractUtilitySpace;

/**
 * Class for keeping the history of bids sent by our opponent, weighted
 * according to the time they were sent.
 * 
 * The sooner they are sent - the higher the weight.
 * 
 * Also tries to estimate, from a given list of acceptable bids, the best one
 * for our opponent, according to the sum of weights for each issue-value pair,
 * in each bid from the range.
 * 
 * Initialization, adding and maintaining the structure are based on work from
 * 2012 competition, by Justin
 */
public class OpponentBidHistory {

	private ArrayList<Bid> bidHistory;

	/**
	 * These arrays map issues to frequency-maps: For each issue, they keep a
	 * map that maps each possible value to the number of times it was offered
	 * by the opponent.
	 */
	private ArrayList<ArrayList<Double>> opponentBidsStatisticsForReal;
	private ArrayList<HashMap<Value, Double>> opponentBidsStatisticsDiscrete;
	private ArrayList<ArrayList<Double>> opponentBidsStatisticsForInteger;

	private int maximumBidsStored = 1000000;
	private Bid bid_maximum_from_opponent;// the bid with maximum utility
											// proposed by the opponent so far.

	public OpponentBidHistory() {
		this.bidHistory = new ArrayList<Bid>();
		opponentBidsStatisticsForReal = new ArrayList<ArrayList<Double>>();
		opponentBidsStatisticsDiscrete = new ArrayList<HashMap<Value, Double>>();
		opponentBidsStatisticsForInteger = new ArrayList<ArrayList<Double>>();
	}

	/**
	 * add a new opponent bid
	 * 
	 * @author Justin
	 */
	public void addBid(Bid bid, AbstractUtilitySpace utilitySpace) {
		if (bidHistory.indexOf(bid) == -1) {
			bidHistory.add(bid);
		}
		try {
			if (bidHistory.size() == 1) {
				this.bid_maximum_from_opponent = bidHistory.get(0);
			} else {
				if (utilitySpace.getUtility(bid) > utilitySpace
						.getUtility(this.bid_maximum_from_opponent)) {
					this.bid_maximum_from_opponent = bid;
				}
			}
		} catch (Exception e) {
			System.out.println("error in addBid method" + e.getMessage());
		}
	}

	public Bid getBestBidInHistory() {
		return this.bid_maximum_from_opponent;
	}

	/**
	 * initialization
	 * 
	 * @author Justin Changed by Eden Erez and Erel Segal haLevi (added weight
	 *         0.0)
	 */
	public void initializeDataStructures(Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();
			for (Issue lIssue : issues) {
				// For each issue, initialize a map from each possible value to
				// integer. The integer is initially 0:

				switch (lIssue.getType()) {

				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					HashMap<Value, Double> discreteIssueValuesMap = new HashMap<Value, Double>();
					for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
						Value v = lIssueDiscrete.getValue(j);
						discreteIssueValuesMap.put(v, 0.0);
					}
					opponentBidsStatisticsDiscrete.add(discreteIssueValuesMap);
					break;

				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					ArrayList<Double> numProposalsPerValue = new ArrayList<Double>();
					int lNumOfPossibleValuesInThisIssue = lIssueReal
							.getNumberOfDiscretizationSteps();
					for (int i = 0; i < lNumOfPossibleValuesInThisIssue; i++) {
						numProposalsPerValue.add(0.0);
					}
					opponentBidsStatisticsForReal.add(numProposalsPerValue);
					break;

				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					ArrayList<Double> numOfValueProposals = new ArrayList<Double>();

					// number of possible value when issue is integer (we should
					// add 1 in order to include all values)
					int lNumOfPossibleValuesForThisIssue = lIssueInteger
							.getUpperBound()
							- lIssueInteger.getLowerBound()
							+ 1;
					for (int i = 0; i < lNumOfPossibleValuesForThisIssue; i++) {
						numOfValueProposals.add(0.0);
					}
					opponentBidsStatisticsForInteger.add(numOfValueProposals);
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("EXCEPTION in initializeDataAtructures");
		}
	}

	/**
	 * This function updates the opponent's Model by calling the
	 * updateStatistics method
	 * 
	 * @author Eden Erez, Erel Segal haLevi
	 */
	public void updateOpponentModel(Bid bidToUpdate, double weight,
			AbstractUtilitySpace utilitySpace) {
		this.addBid(bidToUpdate, utilitySpace);
		if (this.bidHistory.size() <= this.maximumBidsStored) {
			this.updateStatistics(bidToUpdate, weight, utilitySpace.getDomain());
		}
	}

	/**
	 * This function updates the statistics of the bids that were received from
	 * the opponent.
	 * 
	 * New algorithm!
	 * 
	 * @author Justin Changed by Eden Erez and Erel Segal haLevi (added weight)
	 * @since 2013-01
	 */
	private void updateStatistics(Bid bidToUpdate, double weight, Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();

			// counters for each type of the issues
			int realIssueIndex = 0;
			int discreteIssueIndex = 0;
			int integerIssueIndex = 0;
			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum);
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscrete == null) {
						System.out
								.println("opponentBidsStatisticsDiscrete is NULL");
					} else if (opponentBidsStatisticsDiscrete
							.get(discreteIssueIndex) != null) {
						double totalWeightPerValue = opponentBidsStatisticsDiscrete
								.get(discreteIssueIndex).get(v);

						totalWeightPerValue += weight;
						opponentBidsStatisticsDiscrete.get(discreteIssueIndex)
								.put(v, totalWeightPerValue);
					}
					discreteIssueIndex++;
					break;

				case REAL:

					IssueReal lIssueReal = (IssueReal) lIssue;
					int lNumOfPossibleRealValues = lIssueReal
							.getNumberOfDiscretizationSteps();
					double lOneStep = (lIssueReal.getUpperBound() - lIssueReal
							.getLowerBound()) / lNumOfPossibleRealValues;
					double first = lIssueReal.getLowerBound();
					double last = lIssueReal.getLowerBound() + lOneStep;
					double valueReal = ((ValueReal) v).getValue();
					boolean found = false;

					for (int i = 0; !found
							&& i < opponentBidsStatisticsForReal.get(
									realIssueIndex).size(); i++) {
						if (valueReal >= first && valueReal <= last) {
							double countPerValue = opponentBidsStatisticsForReal
									.get(realIssueIndex).get(i);

							countPerValue += weight;

							opponentBidsStatisticsForReal.get(realIssueIndex)
									.set(i, countPerValue);
							found = true;
						}
						first = last;
						last = last + lOneStep;
					}
					// If no matching value was found, update the last cell
					if (found == false) {
						int i = opponentBidsStatisticsForReal.get(
								realIssueIndex).size() - 1;
						double countPerValue = opponentBidsStatisticsForReal
								.get(realIssueIndex).get(i);

						countPerValue += weight;

						opponentBidsStatisticsForReal.get(realIssueIndex).set(
								i, countPerValue);
					}
					realIssueIndex++;
					break;

				case INTEGER:

					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int valueInteger = ((ValueInteger) v).getValue();

					int valueIndex = valueInteger
							- lIssueInteger.getLowerBound(); // For ex.
																// LowerBound
																// index is 0,
																// and the lower
																// bound is 2,
																// the value is
																// 4, so the
																// index of 4
																// would be 2
																// which is
																// exactly 4-2
					double countPerValue = opponentBidsStatisticsForInteger
							.get(integerIssueIndex).get(valueIndex);
					countPerValue += weight;
					opponentBidsStatisticsForInteger.get(integerIssueIndex)
							.set(valueIndex, countPerValue);
					integerIssueIndex++;
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in updateStatistics: "
					+ e.getMessage());
		}
	}

	/**
	 * choose a bid which is optimal for the opponent among a set of candidate
	 * bids.
	 * 
	 * New algorithm!
	 * 
	 * @author Eden Erez, Erel Segal haLevi
	 * @since 2013-01
	 */
	public Bid ChooseBid(List<Bid> candidateBids, Domain domain)
			throws Exception {
		if (candidateBids.isEmpty()) {
			System.out.println("test");
		}
		int indexOfBestCandidateBid = -1;
		List<Issue> issues = domain.getIssues();
		int realIndex = 0;
		int discreteIndex = 0;
		int integerIndex = 0;
		double maxTotalWeightOfAllValuesInCandidateBids = 0;
		for (int iCandidateBid = 0; iCandidateBid < candidateBids.size(); iCandidateBid++) {
			double totalWeightOfAllValuesInCurrentCandidateBid = 0;
			realIndex = discreteIndex = integerIndex = 0;
			for (int iIssue = 0; iIssue < issues.size(); iIssue++) {
				Value valueInCurrentIssueInCurrentCandidateBid = candidateBids
						.get(iCandidateBid).getValue(
								issues.get(iIssue).getNumber());
				switch (issues.get(iIssue).getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscrete == null) {
						System.out
								.println("opponentBidsStatisticsDiscrete is NULL");
					} else if (opponentBidsStatisticsDiscrete
							.get(discreteIndex) != null) {
						double totalWeightPerValue = opponentBidsStatisticsDiscrete
								.get(discreteIndex)
								.get(valueInCurrentIssueInCurrentCandidateBid);
						totalWeightOfAllValuesInCurrentCandidateBid += totalWeightPerValue;
					}
					discreteIndex++;
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) issues.get(iIssue);
					int lNumOfPossibleRealValues = lIssueReal
							.getNumberOfDiscretizationSteps();
					double lOneStep = (lIssueReal.getUpperBound() - lIssueReal
							.getLowerBound()) / lNumOfPossibleRealValues;
					double first = lIssueReal.getLowerBound();
					double last = lIssueReal.getLowerBound() + lOneStep;
					double valueReal = ((ValueReal) valueInCurrentIssueInCurrentCandidateBid)
							.getValue();
					boolean found = false;
					for (int k = 0; !found
							&& k < opponentBidsStatisticsForReal.get(realIndex)
									.size(); k++) {
						if (valueReal >= first && valueReal <= last) {
							double totalWeightPerValue = opponentBidsStatisticsForReal
									.get(realIndex).get(k);
							totalWeightOfAllValuesInCurrentCandidateBid += totalWeightPerValue;
							found = true;
						}
						first = last;
						last = last + lOneStep;
					}
					if (found == false) {
						int k = opponentBidsStatisticsForReal.get(realIndex)
								.size() - 1;
						double totalWeightPerValue = opponentBidsStatisticsForReal
								.get(realIndex).get(k);
						totalWeightOfAllValuesInCurrentCandidateBid += totalWeightPerValue;
					}
					realIndex++;
					break;

				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) issues
							.get(iIssue);
					int valueInteger = ((ValueInteger) valueInCurrentIssueInCurrentCandidateBid)
							.getValue();
					int valueIndex = valueInteger
							- lIssueInteger.getLowerBound(); // For ex.
																// LowerBound
																// index is 0,
																// and the lower
																// bound is 2,
																// the value is
																// 4, so the
																// index of 4
																// would be 2
																// which is
																// exactly 4-2
					double totalWeightPerValue = opponentBidsStatisticsForInteger
							.get(integerIndex).get(valueIndex);
					totalWeightOfAllValuesInCurrentCandidateBid += totalWeightPerValue;
					integerIndex++;
					break;
				}
			}
			if (totalWeightOfAllValuesInCurrentCandidateBid > maxTotalWeightOfAllValuesInCandidateBids) {// choose
																											// the
																											// bid
																											// with
																											// the
																											// maximum
																											// maxValue
				maxTotalWeightOfAllValuesInCandidateBids = totalWeightOfAllValuesInCurrentCandidateBid;
				indexOfBestCandidateBid = iCandidateBid;
			}
		}
		// System.out.println("indexOfBestCandidateBid: " +
		// indexOfBestCandidateBid);
		if (indexOfBestCandidateBid == -1)
			return null;
		return candidateBids.get(indexOfBestCandidateBid);
	}

	/**
	 * @return the number of bids - without duplicates
	 */
	public int getNumberOfDistinctBids() {
		return bidHistory.size();
	}

}
