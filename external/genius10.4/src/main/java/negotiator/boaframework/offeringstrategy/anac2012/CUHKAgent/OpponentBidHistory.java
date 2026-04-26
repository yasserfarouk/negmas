package negotiator.boaframework.offeringstrategy.anac2012.CUHKAgent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Operations related with the opponent's bid history. The opponent's first 100
 * unique bids are remembered.
 * 
 * @author Justin
 */
public class OpponentBidHistory {

	private ArrayList<Bid> bidHistory;
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForReal;
	private ArrayList<HashMap<Value, Integer>> opponentBidsStatisticsDiscrete;
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForInteger;
	private int maximumBidsStored = 100;
	private HashMap<Bid, Integer> bidCounter = new HashMap<Bid, Integer>();
	private Bid bid_maximum_from_opponent;// the bid with maximum utility
											// proposed by the opponent so far.
	private OpponentModel opponentModel;
	private OMStrategy oms;
	private AdditiveUtilitySpace utilitySpace;

	public OpponentBidHistory(OpponentModel opponentModel, OMStrategy oms,
			AdditiveUtilitySpace utilitySpace) {
		this.utilitySpace = utilitySpace;
		this.opponentModel = opponentModel;
		this.oms = oms;
		this.bidHistory = new ArrayList<Bid>();
		opponentBidsStatisticsForReal = new ArrayList<ArrayList<Integer>>();
		opponentBidsStatisticsDiscrete = new ArrayList<HashMap<Value, Integer>>();
		opponentBidsStatisticsForInteger = new ArrayList<ArrayList<Integer>>();
	}

	public void addBid(Bid bid, AdditiveUtilitySpace utilitySpace) {
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
	 */
	public void initializeDataStructures(Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					HashMap<Value, Integer> discreteIssueValuesMap = new HashMap<Value, Integer>();
					for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
						Value v = lIssueDiscrete.getValue(j);
						discreteIssueValuesMap.put(v, 0);
					}
					opponentBidsStatisticsDiscrete.add(discreteIssueValuesMap);
					break;

				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					ArrayList<Integer> numProposalsPerValue = new ArrayList<Integer>();
					int lNumOfPossibleValuesInThisIssue = lIssueReal
							.getNumberOfDiscretizationSteps();
					for (int i = 0; i < lNumOfPossibleValuesInThisIssue; i++) {
						numProposalsPerValue.add(0);
					}
					opponentBidsStatisticsForReal.add(numProposalsPerValue);
					break;

				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					ArrayList<Integer> numOfValueProposals = new ArrayList<Integer>();

					// number of possible value when issue is integer (we should
					// add 1 in order to include all values)
					int lNumOfPossibleValuesForThisIssue = lIssueInteger
							.getUpperBound()
							- lIssueInteger.getLowerBound()
							+ 1;
					for (int i = 0; i < lNumOfPossibleValuesForThisIssue; i++) {
						numOfValueProposals.add(0);
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
	 */
	public void updateOpponentModel(Bid bidToUpdate, Domain domain,
			AdditiveUtilitySpace utilitySpace) {
		this.addBid(bidToUpdate, utilitySpace);

		if (bidCounter.get(bidToUpdate) == null) {
			bidCounter.put(bidToUpdate, 1);
		} else {
			int counter = bidCounter.get(bidToUpdate);
			counter++;
			bidCounter.put(bidToUpdate, counter);
		}

		if (this.bidHistory.size() <= this.maximumBidsStored) {
			this.updateStatistics(bidToUpdate, false, domain);
		}
	}

	/**
	 * This function updates the statistics of the bids that were received from
	 * the opponent.
	 */
	private void updateStatistics(Bid bidToUpdate, boolean toRemove,
			Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();

			// counters for each type of the issues
			int realIndex = 0;
			int discreteIndex = 0;
			int integerIndex = 0;
			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum);
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscrete == null) {
						System.out
								.println("opponentBidsStatisticsDiscrete is NULL");
					} else if (opponentBidsStatisticsDiscrete
							.get(discreteIndex) != null) {
						int counterPerValue = opponentBidsStatisticsDiscrete
								.get(discreteIndex).get(v);
						if (toRemove) {
							counterPerValue--;
						} else {
							counterPerValue++;
						}
						opponentBidsStatisticsDiscrete.get(discreteIndex).put(
								v, counterPerValue);
					}
					discreteIndex++;
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
							&& i < opponentBidsStatisticsForReal.get(realIndex)
									.size(); i++) {
						if (valueReal >= first && valueReal <= last) {
							int countPerValue = opponentBidsStatisticsForReal
									.get(realIndex).get(i);
							if (toRemove) {
								countPerValue--;
							} else {
								countPerValue++;
							}

							opponentBidsStatisticsForReal.get(realIndex).set(i,
									countPerValue);
							found = true;
						}
						first = last;
						last = last + lOneStep;
					}
					// If no matching value was found, receiveMessage the last
					// cell
					if (found == false) {
						int i = opponentBidsStatisticsForReal.get(realIndex)
								.size() - 1;
						int countPerValue = opponentBidsStatisticsForReal.get(
								realIndex).get(i);
						if (toRemove) {
							countPerValue--;
						} else {
							countPerValue++;
						}

						opponentBidsStatisticsForReal.get(realIndex).set(i,
								countPerValue);
					}
					realIndex++;
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
					int countPerValue = opponentBidsStatisticsForInteger.get(
							integerIndex).get(valueIndex);
					if (toRemove) {
						countPerValue--;
					} else {
						countPerValue++;
					}
					opponentBidsStatisticsForInteger.get(integerIndex).set(
							valueIndex, countPerValue);
					integerIndex++;
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in updateStatistics: "
					+ e.getMessage());
		}
	}

	// choose a bid which is optimal for the opponent among a set of candidate
	// bids.

	public Bid ChooseBid(List<Bid> candidateBids, Domain domain) {

		int upperSearchLimit = 200;// 100;

		int maxIndex = -1;
		Random ran = new Random();
		List<Issue> issues = domain.getIssues();
		int maxFrequency = 0;
		int realIndex = 0;
		int discreteIndex = 0;
		int integerIndex = 0;

		if (candidateBids.size() >= upperSearchLimit) {
			List<Bid> bids = new ArrayList<Bid>();
			for (int i = 0; i < upperSearchLimit; i++) {
				int issueIndex = ran.nextInt(candidateBids.size());
				bids.add(candidateBids.get(issueIndex));
			}
			candidateBids = bids;
		}

		if (!(opponentModel instanceof NoModel)) {
			List<BidDetails> bids = new ArrayList<BidDetails>();
			for (Bid bid : candidateBids) {
				try {
					bids.add(new BidDetails(bid, utilitySpace.getUtility(bid)));
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			Bid bid = oms.getBid(bids).getBid();
			return bid;
		} else {
			// this whole block of code is to find the best bid
			try {

				for (int i = 0; i < candidateBids.size(); i++) {
					int maxValue = 0;
					realIndex = discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(i).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscrete
									.get(discreteIndex) != null) {
								int counterPerValue = opponentBidsStatisticsDiscrete
										.get(discreteIndex).get(v);
								maxValue += counterPerValue;
							}
							discreteIndex++;
							break;
						case REAL:
							IssueReal lIssueReal = (IssueReal) issues.get(j);
							int lNumOfPossibleRealValues = lIssueReal
									.getNumberOfDiscretizationSteps();
							double lOneStep = (lIssueReal.getUpperBound() - lIssueReal
									.getLowerBound())
									/ lNumOfPossibleRealValues;
							double first = lIssueReal.getLowerBound();
							double last = lIssueReal.getLowerBound() + lOneStep;
							double valueReal = ((ValueReal) v).getValue();
							boolean found = false;
							for (int k = 0; !found
									&& k < opponentBidsStatisticsForReal.get(
											realIndex).size(); k++) {
								if (valueReal >= first && valueReal <= last) {
									int counterPerValue = opponentBidsStatisticsForReal
											.get(realIndex).get(k);
									maxValue += counterPerValue;
									found = true;
								}
								first = last;
								last = last + lOneStep;
							}
							if (found == false) {
								int k = opponentBidsStatisticsForReal.get(
										realIndex).size() - 1;
								int counterPerValue = opponentBidsStatisticsForReal
										.get(realIndex).get(k);
								maxValue += counterPerValue;
							}
							realIndex++;
							break;

						case INTEGER:
							IssueInteger lIssueInteger = (IssueInteger) issues
									.get(j);
							int valueInteger = ((ValueInteger) v).getValue();
							int valueIndex = valueInteger
									- lIssueInteger.getLowerBound(); // For ex.
																		// LowerBound
																		// index
																		// is 0,
																		// and
																		// the
																		// lower
																		// bound
																		// is 2,
																		// the
																		// value
																		// is 4,
																		// so
																		// the
																		// index
																		// of 4
																		// would
																		// be 2
																		// which
																		// is
																		// exactly
																		// 4-2
							int counterPerValue = opponentBidsStatisticsForInteger
									.get(integerIndex).get(valueIndex);
							maxValue += counterPerValue;
							integerIndex++;
							break;
						}
					}
					if (maxValue > maxFrequency) {// choose the bid with the
													// maximum maxValue
						maxFrequency = maxValue;
						maxIndex = i;
					} else if (maxValue == maxFrequency) {// random exploration
						if (ran.nextDouble() < 0.5) {
							maxFrequency = maxValue;
							maxIndex = i;
						}
					}
				}

			} catch (Exception e) {
				System.out.println("Exception in choosing a bid");
				System.out.println(e.getMessage() + "---" + discreteIndex);
			}
			if (maxIndex == -1) {
				return candidateBids.get(ran.nextInt(candidateBids.size()));
			} else {
				// here we adopt the random exploration mechanism
				if (ran.nextDouble() < 0.95) {
					return candidateBids.get(maxIndex);
				} else {
					return candidateBids.get(ran.nextInt(candidateBids.size()));
				}
			}
		}
	}

	/*
	 * return the best bid from the opponent's bidding history
	 */

	public Bid chooseBestFromHistory(AdditiveUtilitySpace utilitySpace) {
		double max = -1;
		Bid maxBid = null;
		try {
			for (Bid bid : bidHistory) {
				if (max < utilitySpace.getUtility(bid)) {
					max = utilitySpace.getUtility(bid);
					maxBid = bid;
				}
			}
		} catch (Exception e) {
			System.out.println("ChooseBestfromhistory exception");
		}
		return maxBid;
	}

	// one way to predict the concession degree of the opponent
	public double concedeDegree(AdditiveUtilitySpace utilitySpace) {
		int numOfBids = bidHistory.size();
		HashMap<Bid, Integer> bidCounter = new HashMap<Bid, Integer>();
		try {
			for (int i = 0; i < numOfBids; i++) {

				if (bidCounter.get(bidHistory.get(i)) == null) {
					bidCounter.put(bidHistory.get(i), 1);
				} else {
					int counter = bidCounter.get(bidHistory.get(i));
					counter++;
					bidCounter.put(bidHistory.get(i), counter);
				}
			}
		} catch (Exception e) {
			System.out.println("ChooseBestfromhistory exception");
		}
		// System.out.println("the opponent's toughness degree is " +
		// bidCounter.size() + " divided by " +
		// utilitySpace.getDomain().getNumberOfPossibleBids());
		return ((double) bidCounter.size() / utilitySpace.getDomain()
				.getNumberOfPossibleBids());
	}

	public int getSize() {
		int numOfBids = bidHistory.size();
		HashMap<Bid, Integer> bidCounter = new HashMap<Bid, Integer>();
		try {
			for (int i = 0; i < numOfBids; i++) {

				if (bidCounter.get(bidHistory.get(i)) == null) {
					bidCounter.put(bidHistory.get(i), 1);
				} else {
					int counter = bidCounter.get(bidHistory.get(i));
					counter++;
					bidCounter.put(bidHistory.get(i), counter);
				}
			}
		} catch (Exception e) {
			System.out.println("getSize exception");
		}
		return bidCounter.size();
	}

	// Another way to predict the opponent's concession degree
	public double getConcessionDegree() {
		int numOfBids = bidHistory.size();
		double numOfDistinctBid = 0;
		int historyLength = 10;
		double concessionDegree = 0;
		// HashMap<Bid, Integer> bidCounter = new HashMap<Bid, Integer>();
		if (numOfBids - historyLength > 0) {
			try {
				for (int j = numOfBids - historyLength; j < numOfBids; j++) {
					if (bidCounter.get(bidHistory.get(j)) == 1) {
						numOfDistinctBid++;
					}
				}
				concessionDegree = Math
						.pow(numOfDistinctBid / historyLength, 2);
			} catch (Exception e) {
				System.out.println("getConcessionDegree exception");
			}
		} else {
			numOfDistinctBid = this.getSize();
			concessionDegree = Math.pow(numOfDistinctBid / historyLength, 2);
		}
		// System.out.println("the history length is" + bidHistory.size() +
		// "concessiondegree is " + concessionDegree);
		return concessionDegree;
	}
}
