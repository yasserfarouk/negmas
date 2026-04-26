package agents.anac.y2015.AgentNeo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.utility.AbstractUtilitySpace;

public class BidOptions {

	private ArrayList<Bid> bidHistoryOpp1;
	private ArrayList<Bid> bidHistoryOpp2;
	private ArrayList<Bid> bidHistoryOpp3;
	private static ArrayList<Bid> bidHistory;

	// private int maximumBidsStored = 100;
	// private HashMap<Bid, Integer> bidCounter = new HashMap<Bid, Integer>();
	int opp1_token_counter = 0;
	int opp2_token_counter = 0;
	int opp3_token_counter = 0;
	int temp_token_counter = 0;
	private Bid bid_maximum;
	private Bid bid_maximum_from_opponent1;
	private Bid bid_maximum_from_opponent2;
	private Bid bid_maximum_from_opponent3;
	private double bid_maximum_utility_from_opponent1;
	private double bid_maximum_utility_from_opponent2;
	private double bid_maximum_utility_from_opponent3;// the bid with maximum
														// utility proposed by
														// the opponent so far.
	private ArrayList<HashMap<Value, Integer>> opponentBidsStatisticsDiscreteOpp1;
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForIntegerOpp1;
	private ArrayList<HashMap<Value, Integer>> opponentBidsStatisticsDiscreteOpp2;
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForIntegerOpp2;
	private ArrayList<HashMap<Value, Integer>> opponentBidsStatisticsDiscreteOpp3;
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForIntegerOpp3;
	HashMap<String, Integer> token_counter = new HashMap<String, Integer>();

	public BidOptions() {
		this.bidHistoryOpp1 = new ArrayList<Bid>();
		this.bidHistoryOpp2 = new ArrayList<Bid>();
		this.bidHistoryOpp3 = new ArrayList<Bid>();
		BidOptions.bidHistory = new ArrayList<Bid>();
		opponentBidsStatisticsDiscreteOpp1 = new ArrayList<HashMap<Value, Integer>>();
		opponentBidsStatisticsForIntegerOpp1 = new ArrayList<ArrayList<Integer>>();
		opponentBidsStatisticsDiscreteOpp2 = new ArrayList<HashMap<Value, Integer>>();
		opponentBidsStatisticsForIntegerOpp2 = new ArrayList<ArrayList<Integer>>();
		opponentBidsStatisticsDiscreteOpp3 = new ArrayList<HashMap<Value, Integer>>();
		opponentBidsStatisticsForIntegerOpp3 = new ArrayList<ArrayList<Integer>>();

	}

	protected void initializeDataStructures(Domain domain) {
		try {

			List<Issue> issues = domain.getIssues();
			System.out.println(issues);
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {

				case DISCRETE:

					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					HashMap<Value, Integer> discreteIssueValuesMap = new HashMap<Value, Integer>();
					for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
						Value v = lIssueDiscrete.getValue(j);
						discreteIssueValuesMap.put(v, 0);
						System.out.println(discreteIssueValuesMap);
					}

					opponentBidsStatisticsDiscreteOpp1
							.add(discreteIssueValuesMap);
					opponentBidsStatisticsDiscreteOpp2
							.add(discreteIssueValuesMap);
					opponentBidsStatisticsDiscreteOpp3
							.add(discreteIssueValuesMap);

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
					opponentBidsStatisticsForIntegerOpp1
							.add(numOfValueProposals);
					opponentBidsStatisticsForIntegerOpp2
							.add(numOfValueProposals);
					opponentBidsStatisticsForIntegerOpp3
							.add(numOfValueProposals);

					break;
				}
			}
		} catch (Exception e) {
			System.out.println("EXCEPTION in initializeDataStructures");
		}
	}

	protected void addBidOpp1(Bid bid, AbstractUtilitySpace utilitySpace) {

		bidHistory.add(bid);
		if (bidHistoryOpp1.indexOf(bid) == -1) { // list does not contain the
													// bid
			bidHistoryOpp1.add(bid);

		}
		try {
			if (bidHistoryOpp1.size() == 1) {
				this.bid_maximum_from_opponent1 = bidHistoryOpp1.get(0);
				this.bid_maximum_utility_from_opponent1 = utilitySpace
						.getUtility(this.bid_maximum_from_opponent1);

			} else {

				this.bid_maximum_from_opponent1 = bid;
				this.bid_maximum_utility_from_opponent1 = utilitySpace
						.getUtility(bid);
			}
		} catch (Exception e) {
			System.out.println("error in addBid method" + e.getMessage());
		}
	}

	protected void addBidOpp2(Bid bid, AbstractUtilitySpace utilitySpace) {

		bidHistory.add(bid);
		if (bidHistoryOpp2.indexOf(bid) == -1) { // list does not contain the
													// bid
			bidHistoryOpp2.add(bid);

		}
		try {
			if (bidHistoryOpp2.size() == 1) {
				this.bid_maximum_from_opponent2 = bidHistoryOpp2.get(0);
				this.bid_maximum_utility_from_opponent2 = utilitySpace
						.getUtility(this.bid_maximum_from_opponent2);

			} else {

				this.bid_maximum_from_opponent2 = bid;
				this.bid_maximum_utility_from_opponent2 = utilitySpace
						.getUtility(bid);

			}
		} catch (Exception e) {
			System.out.println("error in addBid method" + e.getMessage());
		}
	}

	protected void addBidOpp3(Bid bid, AbstractUtilitySpace utilitySpace) {

		bidHistory.add(bid);
		if (bidHistoryOpp3.indexOf(bid) == -1) { // list does not contain the
													// bid
			bidHistoryOpp3.add(bid);

		}

		try {
			if (bidHistoryOpp3.size() == 1) {
				this.bid_maximum_from_opponent3 = bidHistoryOpp3.get(0);
				this.bid_maximum_utility_from_opponent3 = utilitySpace
						.getUtility(this.bid_maximum_from_opponent3);

			} else {

				this.bid_maximum_from_opponent3 = bid;
				this.bid_maximum_utility_from_opponent3 = utilitySpace
						.getUtility(bid);

			}
		} catch (Exception e) {
			System.out.println("error in addBid method" + e.getMessage());
		}
	}

	protected void addOpponentBid(Bid bidToUpdate, Domain domain,
			AbstractUtilitySpace utilitySpace, Object IDOfOpponent) {
		String OppID = IDOfOpponent.toString();

		if (OppID.equals("Party 1")) {
			this.addBidOpp1(bidToUpdate, utilitySpace);
			this.updateStatisticsOpp1(bidToUpdate, false, domain);
		}
		if (OppID.equals("Party 2")) {
			this.addBidOpp2(bidToUpdate, utilitySpace);
			this.updateStatisticsOpp2(bidToUpdate, false, domain);
		}
		if (OppID.equals("Party 3")) {
			this.addBidOpp3(bidToUpdate, utilitySpace);
			this.updateStatisticsOpp3(bidToUpdate, false, domain);
		}
	}

	private void updateStatisticsOpp1(Bid bidToUpdate, boolean toRemove,
			Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();

			// counters for each type of the issues

			int discreteIndex = 0;
			int integerIndex = 0;

			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum); // v is the reference
															// key
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscreteOpp1 == null) {
						System.out
								.println("opponentBidsStatisticsDiscreteOpp1 is NULL");
					} else if (opponentBidsStatisticsDiscreteOpp1
							.get(discreteIndex) != null) {
						int ValueofCounter = opponentBidsStatisticsDiscreteOpp1
								.get(discreteIndex).get(v);
						if (toRemove) {
							ValueofCounter--;
						} else {
							ValueofCounter++;
						}
						opponentBidsStatisticsDiscreteOpp1.get(discreteIndex)
								.put(v, ValueofCounter);
					}
					discreteIndex++;
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
					int countPerValue = opponentBidsStatisticsForIntegerOpp1
							.get(integerIndex).get(valueIndex);
					if (toRemove) {
						countPerValue--;
					} else {
						countPerValue++;
					}
					opponentBidsStatisticsForIntegerOpp1.get(integerIndex).set(
							valueIndex, countPerValue);
					integerIndex++;
					break;
				default:
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in updateStatistics1: "
					+ e.getMessage());
		}
	}

	private void updateStatisticsOpp2(Bid bidToUpdate, boolean toRemove,
			Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();

			// counters for each type of the issues

			int discreteIndex = 0;
			int integerIndex = 0;

			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum); // v is the reference
															// key
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscreteOpp2 == null) {
						System.out
								.println("opponentBidsStatisticsDiscreteOpp2 is NULL");
					} else if (opponentBidsStatisticsDiscreteOpp2
							.get(discreteIndex) != null) {
						int ValueofCounter = opponentBidsStatisticsDiscreteOpp2
								.get(discreteIndex).get(v);
						if (toRemove) {
							ValueofCounter--;
						} else {
							ValueofCounter++;
						}
						opponentBidsStatisticsDiscreteOpp2.get(discreteIndex)
								.put(v, ValueofCounter);
					}
					discreteIndex++;
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
					int countPerValue = opponentBidsStatisticsForIntegerOpp2
							.get(integerIndex).get(valueIndex);
					if (toRemove) {
						countPerValue--;
					} else {
						countPerValue++;
					}
					opponentBidsStatisticsForIntegerOpp2.get(integerIndex).set(
							valueIndex, countPerValue);
					integerIndex++;
					break;
				default:
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in updateStatistics2: "
					+ e.getMessage());
		}
	}

	private void updateStatisticsOpp3(Bid bidToUpdate, boolean toRemove,
			Domain domain) {
		try {
			List<Issue> issues = domain.getIssues();

			// counters for each type of the issues

			int discreteIndex = 0;
			int integerIndex = 0;

			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum); // v is the reference
															// key
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscreteOpp3 == null) {
						System.out
								.println("opponentBidsStatisticsDiscreteOpp3 is NULL");
					} else if (opponentBidsStatisticsDiscreteOpp3
							.get(discreteIndex) != null) {
						int ValueofCounter = opponentBidsStatisticsDiscreteOpp3
								.get(discreteIndex).get(v);
						if (toRemove) {
							ValueofCounter--;
						} else {
							ValueofCounter++;
						}
						opponentBidsStatisticsDiscreteOpp3.get(discreteIndex)
								.put(v, ValueofCounter);
					}
					discreteIndex++;
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
					int countPerValue = opponentBidsStatisticsForIntegerOpp3
							.get(integerIndex).get(valueIndex);
					if (toRemove) {
						countPerValue--;
					} else {
						countPerValue++;
					}
					opponentBidsStatisticsForIntegerOpp3.get(integerIndex).set(
							valueIndex, countPerValue);
					integerIndex++;
					break;
				default:
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in updateStatistics3: "
					+ e.getMessage());
		}
	}

	protected Bid getBestBidInHistory() {

		if (token_counter.get("opponent 1") > token_counter.get("opponent 2")
				&& token_counter.get("opponent 1") > token_counter
						.get("opponent 3")) {

			bid_maximum = this.bid_maximum_from_opponent1;

		} else if (token_counter.get("opponent 2") > token_counter
				.get("opponent 1")
				&& token_counter.get("opponent 2") > token_counter
						.get("opponent 3")) {

			bid_maximum = this.bid_maximum_from_opponent2;

		} else if (token_counter.get("opponent 3") > token_counter
				.get("opponent 2")
				&& token_counter.get("opponent 3") > token_counter
						.get("opponent 1")) {

			bid_maximum = this.bid_maximum_from_opponent3;

		}
		// Determine who is the opponent I am siding with
		return bid_maximum;
	}

	enum maxType {
		A, B, C
	};

	protected int SimilartoOpponent(int value) {

		double a = this.bid_maximum_utility_from_opponent1;
		double b = this.bid_maximum_utility_from_opponent2;
		double c = this.bid_maximum_utility_from_opponent3;

		maxType max = maxType.A;

		// System.out.println(a);
		// System.out.println(b);
		// System.out.println(c);

		if (b > a && b > c) {
			max = maxType.B;
		}
		if (c > b && c > a) {
			max = maxType.C;
		}

		// System.out.println(max);

		if (token_counter.isEmpty()) {
			token_counter.put("opponent 1", 0);
			token_counter.put("opponent 2", 0);
			token_counter.put("opponent 3", 0);
		}

		System.out.println(token_counter);
		System.out.println(max);
		switch (max) {

		case A:

			temp_token_counter = token_counter.get("opponent 1") + 1;
			token_counter.put("opponent 1", temp_token_counter);
			break;

		case B:

			temp_token_counter = token_counter.get("opponent 2") + 1;
			token_counter.put("opponent 2", temp_token_counter);
			break;

		case C:

			temp_token_counter = token_counter.get("opponent 3") + 1;
			token_counter.put("opponent 3", temp_token_counter);
			break;
		}

		System.out.println(token_counter);

		if (token_counter.get("opponent 1") >= token_counter.get("opponent 2")
				&& token_counter.get("opponent 1") >= token_counter
						.get("opponent 3")) {

			// System.out.println("A");
			value = 1;
		} else if (token_counter.get("opponent 2") >= token_counter
				.get("opponent 1")
				&& token_counter.get("opponent 2") >= token_counter
						.get("opponent 3")) {
			// System.out.println("B");
			value = 2;
		} else if (token_counter.get("opponent 3") >= token_counter
				.get("opponent 2")
				&& token_counter.get("opponent 3") >= token_counter
						.get("opponent 1")) {
			// System.out.println("C");
			value = 3;
		}
		// System.out.println(value);

		return value;

	}

	protected Bid ChooseBid1(List<Bid> candidateBids, Domain domain) {
		int upperSearchLimit = 200;// 100;
		if (candidateBids.isEmpty()) {
			System.out.println("test");
		}
		int maxIndex = -1;
		Random ran = new Random();
		List<Issue> issues = domain.getIssues();
		int MaximumFreq = 0;
		int discreteIndex = 0;
		int integerIndex = 0;
		try {
			if (candidateBids.size() < upperSearchLimit) {
				for (int i = 0; i < candidateBids.size(); i++) {
					int MaximumCount = 0;
					discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(i).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscreteOpp1 == null) {
								System.out
										.println("opponentBidsStatisticsDiscrete is NULL");
							} else if (opponentBidsStatisticsDiscreteOpp1
									.get(discreteIndex) != null) {
								int ValueofCounter = opponentBidsStatisticsDiscreteOpp1
										.get(discreteIndex).get(v);
								MaximumCount += ValueofCounter;
							}
							discreteIndex++;
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
							int ValueofCounter = opponentBidsStatisticsForIntegerOpp1
									.get(integerIndex).get(valueIndex);
							MaximumCount += ValueofCounter;
							integerIndex++;
							break;
						}
					}
					if (MaximumCount > MaximumFreq) {// choose the bid with the
														// maximum MaximumCount
						MaximumFreq = MaximumCount;
						maxIndex = i;
					} else if (MaximumCount == MaximumFreq) {// random
																// exploration
						if (ran.nextDouble() < 0.5) {
							MaximumFreq = MaximumCount;
							maxIndex = i;
						}
					}
				}

			} else {// only evaluate the upperSearchLimit number of bids
				for (int i = 0; i < upperSearchLimit; i++) {
					int MaximumCount = 0;
					int issueIndex = ran.nextInt(candidateBids.size());
					discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(issueIndex).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscreteOpp1 == null) {
								System.out
										.println("opponentBidsStatisticsDiscrete is NULL");
							} else if (opponentBidsStatisticsDiscreteOpp1
									.get(discreteIndex) != null) {
								int ValueofCounter = opponentBidsStatisticsDiscreteOpp1
										.get(discreteIndex).get(v);
								MaximumCount += ValueofCounter;
							}
							discreteIndex++;
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
							int ValueofCounter = opponentBidsStatisticsForIntegerOpp1
									.get(integerIndex).get(valueIndex);
							MaximumCount += ValueofCounter;
							integerIndex++;
							break;
						}
					}
					if (MaximumCount > MaximumFreq) {// choose the bid with the
														// maximum MaximumCount
						MaximumFreq = MaximumCount;
						maxIndex = i;
					} else if (MaximumCount == MaximumFreq) {// random
																// exploration
						if (ran.nextDouble() < 0.5) {
							MaximumFreq = MaximumCount;
							maxIndex = i;
						}
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

	protected Bid ChooseBid2(List<Bid> candidateBids, Domain domain) {
		int upperSearchLimit = 200;// 100;
		if (candidateBids.isEmpty()) {
			System.out.println("test");
		}
		int maxIndex = -1;
		Random ran = new Random();
		List<Issue> issues = domain.getIssues();
		int MaximumFreq = 0;
		int discreteIndex = 0;
		int integerIndex = 0;
		try {
			if (candidateBids.size() < upperSearchLimit) {
				for (int i = 0; i < candidateBids.size(); i++) {
					int MaximumCount = 0;
					discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(i).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscreteOpp2 == null) {
								System.out
										.println("opponentBidsStatisticsDiscrete is NULL");
							} else if (opponentBidsStatisticsDiscreteOpp2
									.get(discreteIndex) != null) {
								int ValueofCounter = opponentBidsStatisticsDiscreteOpp2
										.get(discreteIndex).get(v);
								MaximumCount += ValueofCounter;
							}
							discreteIndex++;
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
							int ValueofCounter = opponentBidsStatisticsForIntegerOpp2
									.get(integerIndex).get(valueIndex);
							MaximumCount += ValueofCounter;
							integerIndex++;
							break;
						}
					}
					if (MaximumCount > MaximumFreq) {// choose the bid with the
														// maximum MaximumCount
						MaximumFreq = MaximumCount;
						maxIndex = i;
					} else if (MaximumCount == MaximumFreq) {// random
																// exploration
						if (ran.nextDouble() < 0.5) {
							MaximumFreq = MaximumCount;
							maxIndex = i;
						}
					}
				}

			} else {// only evaluate the upperSearchLimit number of bids
				for (int i = 0; i < upperSearchLimit; i++) {
					int MaximumCount = 0;
					int issueIndex = ran.nextInt(candidateBids.size());
					discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(issueIndex).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscreteOpp2 == null) {
								System.out
										.println("opponentBidsStatisticsDiscrete is NULL");
							} else if (opponentBidsStatisticsDiscreteOpp2
									.get(discreteIndex) != null) {
								int ValueofCounter = opponentBidsStatisticsDiscreteOpp2
										.get(discreteIndex).get(v);
								MaximumCount += ValueofCounter;
							}
							discreteIndex++;
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
							int ValueofCounter = opponentBidsStatisticsForIntegerOpp2
									.get(integerIndex).get(valueIndex);
							MaximumCount += ValueofCounter;
							integerIndex++;
							break;
						}
					}
					if (MaximumCount > MaximumFreq) {// choose the bid with the
														// maximum MaximumCount
						MaximumFreq = MaximumCount;
						maxIndex = i;
					} else if (MaximumCount == MaximumFreq) {// random
																// exploration
						if (ran.nextDouble() < 0.5) {
							MaximumFreq = MaximumCount;
							maxIndex = i;
						}
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

	protected Bid ChooseBid3(List<Bid> candidateBids, Domain domain) {
		int upperSearchLimit = 200;// 100;
		if (candidateBids.isEmpty()) {
			System.out.println("test");
		}
		int maxIndex = -1;
		Random ran = new Random();
		List<Issue> issues = domain.getIssues();
		int MaximumFreq = 0;
		int discreteIndex = 0;
		int integerIndex = 0;
		try {
			if (candidateBids.size() < upperSearchLimit) {
				for (int i = 0; i < candidateBids.size(); i++) {
					int MaximumCount = 0;
					discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(i).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscreteOpp3 == null) {
								System.out
										.println("opponentBidsStatisticsDiscrete is NULL");
							} else if (opponentBidsStatisticsDiscreteOpp3
									.get(discreteIndex) != null) {
								int ValueofCounter = opponentBidsStatisticsDiscreteOpp3
										.get(discreteIndex).get(v);
								MaximumCount += ValueofCounter;
							}
							discreteIndex++;
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
							int ValueofCounter = opponentBidsStatisticsForIntegerOpp3
									.get(integerIndex).get(valueIndex);
							MaximumCount += ValueofCounter;
							integerIndex++;
							break;
						}
					}
					if (MaximumCount > MaximumFreq) {// choose the bid with the
														// maximum MaximumCount
						MaximumFreq = MaximumCount;
						maxIndex = i;
					} else if (MaximumCount == MaximumFreq) {// random
																// exploration
						if (ran.nextDouble() < 0.5) {
							MaximumFreq = MaximumCount;
							maxIndex = i;
						}
					}
				}

			} else {// only evaluate the upperSearchLimit number of bids
				for (int i = 0; i < upperSearchLimit; i++) {
					int MaximumCount = 0;
					int issueIndex = ran.nextInt(candidateBids.size());
					discreteIndex = integerIndex = 0;
					for (int j = 0; j < issues.size(); j++) {
						Value v = candidateBids.get(issueIndex).getValue(
								issues.get(j).getNumber());
						switch (issues.get(j).getType()) {
						case DISCRETE:
							if (opponentBidsStatisticsDiscreteOpp3 == null) {
								System.out
										.println("opponentBidsStatisticsDiscrete is NULL");
							} else if (opponentBidsStatisticsDiscreteOpp3
									.get(discreteIndex) != null) {
								int ValueofCounter = opponentBidsStatisticsDiscreteOpp3
										.get(discreteIndex).get(v);
								MaximumCount += ValueofCounter;
							}
							discreteIndex++;
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
							int ValueofCounter = opponentBidsStatisticsForIntegerOpp3
									.get(integerIndex).get(valueIndex);
							MaximumCount += ValueofCounter;
							integerIndex++;
							break;
						}
					}
					if (MaximumCount > MaximumFreq) {// choose the bid with the
														// maximum MaximumCount
						MaximumFreq = MaximumCount;
						maxIndex = i;
					} else if (MaximumCount == MaximumFreq) {// random
																// exploration
						if (ran.nextDouble() < 0.5) {
							MaximumFreq = MaximumCount;
							maxIndex = i;
						}
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

	protected static Bid getLastBid() {
		if (bidHistory.size() >= 1) {
			return bidHistory.get(bidHistory.size() - 1);
		} else {
			return null;
		}
	}
}