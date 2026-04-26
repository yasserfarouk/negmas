package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2011.BRAMAgentSAS;

/**
 * This is the decoupled Offering Strategy for BRAMAgent (ANAC2011) The code was
 * taken from the ANAC2011 BRAMAgent and adapted to work within the BOA
 * framework
 * 
 * For the opponent model extension a range of bids is found near the target
 * utility. The opponent model strategy uses the OM to select a bid from this
 * range of bids.
 * 
 * DEFAULT OM: None
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class BRAMAgent_Offering extends OfferingStrategy {

	/* FINAL VARIABLES */
	private final int TIME_TO_CREATE_BIDS_ARRAY = 2000;// The time that we
														// allocate to creating
														// the bids array
	private final double FREQUENCY_OF_PROPOSAL = 0.2;// If the frequency of the
														// proposal is larger
														// than this variable
														// than we won't propose
														// it

	// The number of opponent's bids that we save in order to learn its
	// preferences
	private final int OPPONENT_ARRAY_SIZE = 10;
	/* MEMBERS */

	private Bid bestBid;// The best bid that our agent offered
	private double maxUtility;// The maximum utility that our agent can get
	private ArrayList<Bid> ourBidsArray;// An Array that contains all the bids
										// that our agent can offer
	private ArrayList<Bid> opponentBidsArray;// An Array that contains the last
												// 10 bids that the opponent
												// agent offered
	private int lastPositionInBidArray;// The position in the bid array of the
										// our agent last offer
	private int[] bidsCountProposalArray;// An array that saves the number of
											// offers that were made per each
											// bid
	private int numOfProposalsFromOurBidsArray;// The number of proposals that
												// were made - NOT including the
												// proposals that were made in
												// the TIME_TO_OFFER_MAX_BID
												// time
	private double threshold;// The threshold - we will accept any offer that
								// its utility is larger than the threshold
	private int randomInterval;
	private int randomOffset;
	/* Data Structures for any type of issue */
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForReal;
	private ArrayList<HashMap<Value, Integer>> opponentBidsStatisticsDiscrete;
	private ArrayList<ArrayList<Integer>> opponentBidsStatisticsForInteger;
	private Bid previousOfferedBid;
	private Random random100;
	private Random random200;
	private Random random300;
	private final boolean TEST_EQUIVALENCE = false;
	private SortedOutcomeSpace outcomespace;
	int round = 0;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public BRAMAgent_Offering() {
	}

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel om, OMStrategy oms, Map<String, Double> parameters)
			throws Exception {
		if (om instanceof DefaultModel) {
			om = new NoModel();
		}
		if (!(opponentModel instanceof NoModel)) {
			outcomespace = new SortedOutcomeSpace(domainKnow.getUtilitySpace());
		}
		initializeAgent(domainKnow, om, oms);
	}

	public void initializeAgent(NegotiationSession negoSession, OpponentModel om, OMStrategy oms) {
		this.negotiationSession = negoSession;
		this.omStrategy = oms;
		ourBidsArray = new ArrayList<Bid>();
		bidsCountProposalArray = null;
		lastPositionInBidArray = 0;
		numOfProposalsFromOurBidsArray = 0;
		randomInterval = 8;
		randomOffset = 4;
		opponentBidsArray = new ArrayList<Bid>();
		opponentModel = om;
		helper = new BRAMAgentSAS(negotiationSession);
		initializeDataStructures();
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
			random300 = new Random(300);
		} else {
			random100 = new Random();
			random200 = new Random();
			random300 = new Random();
		}

		try {
			bestBid = negotiationSession.getUtilitySpace().getMaxUtilityBid();
			maxUtility = negotiationSession.getUtilitySpace().getUtilityWithDiscount(bestBid,
					negotiationSession.getTimeline());
			ourBidsArray.add(bestBid);// The offer with the maximum utility will
										// be offered at the beginning
			threshold = maxUtility;
			previousOfferedBid = bestBid;

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public BidDetails determineNextBid() {
		round++;
		threshold = ((BRAMAgentSAS) helper).getNewThreshold(ourBidsArray.get(ourBidsArray.size() - 1), bestBid);// Update
																												// the
																												// threshold
																												// according
																												// to
																												// the
																												// discount
																												// factor
		BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();

		if (opponentBid == null) {
			nextBid = negotiationSession.getMaxBidinDomain();
		} else {
			try {
				Bid bidToRemove = null;

				Bid bidToOffer = null;

				if (opponentBidsArray.size() < OPPONENT_ARRAY_SIZE) {// In this
																		// phase
																		// we
																		// are
																		// gathering
																		// information
																		// about
																		// the
																		// bids
																		// that
																		// the
																		// opponent
																		// is
																		// offering
					opponentBidsArray.add(opponentBid.getBid());
					updateStatistics(opponentBid.getBid(), false);
					bidToOffer = bestBid;
				} else {
					// Remove the oldest bid and receiveMessage the statistics
					bidToRemove = opponentBidsArray.get(0);
					updateStatistics(bidToRemove, true);

					opponentBidsArray.remove(0);
					// Add the new bid of the opponent and receiveMessage the
					// statistics
					opponentBidsArray.add(opponentBid.getBid());
					updateStatistics(opponentBid.getBid(), false);
					if (opponentModel instanceof NoModel) {
						bidToOffer = getBidToOffer();
					} else {
						threshold = ((BRAMAgentSAS) helper).getNewThreshold(ourBidsArray.get(ourBidsArray.size() - 1),
								getBidToOffer());// Update the threshold
													// according to the discount
													// factor
						bidToOffer = omStrategy.getBid(outcomespace, threshold).getBid();
					}
				}

				nextBid = new BidDetails(bidToOffer, negotiationSession.getUtilitySpace().getUtility(bidToOffer),
						negotiationSession.getTime());

			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return nextBid;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	/**
	 * This function updates the statistics of the bids that were received from
	 * the opponent
	 * 
	 * @param bidToUpdate
	 *            - the bid that we want to receiveMessage its statistics
	 * @param toRemove
	 *            - flag that indicates if we removing (or adding) a bid to (or
	 *            from) the statistics
	 */
	private void updateStatistics(Bid bidToUpdate, boolean toRemove) {
		try {
			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

			// counters for each type of issue
			int realIndex = 0;
			int discreteIndex = 0;
			int integerIndex = 0;

			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum);
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscrete != null
							&& opponentBidsStatisticsDiscrete.get(discreteIndex) != null) {
						int counterPerValue = opponentBidsStatisticsDiscrete.get(discreteIndex).get(v);
						if (toRemove)
							counterPerValue--;
						else
							counterPerValue++;
						opponentBidsStatisticsDiscrete.get(discreteIndex).put(v, counterPerValue);
					}
					discreteIndex++;
					break;

				case REAL:

					IssueReal lIssueReal = (IssueReal) lIssue;
					int lNumOfPossibleRealValues = lIssueReal.getNumberOfDiscretizationSteps();
					double lOneStep = (lIssueReal.getUpperBound() - lIssueReal.getLowerBound())
							/ lNumOfPossibleRealValues;
					double first = lIssueReal.getLowerBound();
					double last = lIssueReal.getLowerBound() + lOneStep;
					double valueReal = ((ValueReal) v).getValue();
					boolean found = false;

					for (int i = 0; !found && i < opponentBidsStatisticsForReal.get(realIndex).size(); i++) {
						if (valueReal >= first && valueReal <= last) {
							int countPerValue = opponentBidsStatisticsForReal.get(realIndex).get(i);
							if (toRemove)
								countPerValue--;
							else
								countPerValue++;

							opponentBidsStatisticsForReal.get(realIndex).set(i, countPerValue);
							found = true;
						}
						first = last;
						last = last + lOneStep;
					}
					// If no matching value was found, receiveMessage the last
					// cell
					if (found == false) {
						int i = opponentBidsStatisticsForReal.get(realIndex).size() - 1;
						int countPerValue = opponentBidsStatisticsForReal.get(realIndex).get(i);
						if (toRemove)
							countPerValue--;
						else
							countPerValue++;

						opponentBidsStatisticsForReal.get(realIndex).set(i, countPerValue);
					}
					realIndex++;
					break;

				case INTEGER:

					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int valueInteger = ((ValueInteger) v).getValue();

					int valueIndex = valueInteger - lIssueInteger.getLowerBound(); // For
																					// ex.
																					// LowerBound
																					// index
																					// is
																					// 0,
																					// and
																					// the
																					// lower
																					// bound
																					// is
																					// 2,
																					// the
																					// value
																					// is
																					// 4,
																					// so
																					// the
																					// index
																					// of
																					// 4
																					// would
																					// be
																					// 2
																					// which
																					// is
																					// exactly
																					// 4-2
					int countPerValue = opponentBidsStatisticsForInteger.get(integerIndex).get(valueIndex);
					if (toRemove)
						countPerValue--;
					else
						countPerValue++;

					opponentBidsStatisticsForInteger.get(integerIndex).set(valueIndex, countPerValue);
					integerIndex++;
					break;
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * This function calculates the bid that the agent offers. If a calculated
	 * bid is close enough to the preferences of the opponent, and its utility
	 * is acceptable by our agent, our agent will offer it. Otherwise, we will
	 * offer a bid from
	 * 
	 * @return
	 */
	private Bid getBidToOffer() {
		Bid bidWithMaxUtility = null;
		try {
			double maxUt = threshold;

			for (int i = 0; i < 10; i++) {
				Bid currBid = createBidByOpponentModeling();
				if (currBid != null) {
					double currUtility = negotiationSession.getUtilitySpace().getUtilityWithDiscount(currBid,
							negotiationSession.getTime());

					if (currUtility > maxUt) {
						maxUt = currUtility;
						bidWithMaxUtility = currBid;
					}
				}

			}
			if (bidWithMaxUtility == null) {
				return getBidFromBidsArray();

			} else {
				return bidWithMaxUtility;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bidWithMaxUtility;
	}

	/**
	 * This function creates random bids that the agent can offer and sorts it
	 * in a descending order.
	 * 
	 * @return
	 */
	private Bid getBidFromBidsArray() {

		if (ourBidsArray.size() == 1) {
			// We get here only at the first time - when we want to build the
			// array
			fillBidsArray(new Date().getTime());
			if (ourBidsArray.size() <= 50) {
				randomInterval = 3;
				randomOffset = 1;
			}
			initializeBidsFrequencyArray();
			Collections.sort(ourBidsArray, new Comparator<Bid>() {
				// @Override
				public int compare(Bid bid1, Bid bid2) {
					// We will sort the array in a descending order
					double utility1 = 0.0;
					double utility2 = 0.0;
					try {
						utility1 = negotiationSession.getUtilitySpace().getUtility(bid1);
						utility2 = negotiationSession.getUtilitySpace().getUtility(bid2);
						if (utility1 > utility2)
							return -1;
						else if (utility1 < utility2)
							return 1;
					} catch (Exception e) {
						e.printStackTrace();
					}
					return 0;
				}
			});
		}

		// We will make an offer
		numOfProposalsFromOurBidsArray++;
		Bid bidToOffer = selectCurrentBidFromOurBidsArray();
		return bidToOffer;
	}

	/**
	 * This function creates a bid according to the preferences of the opponent.
	 * Meaning, we assume that if the opponent insisted on some value of an
	 * issue, it's probably important to it.
	 * 
	 * @return
	 */
	private Bid createBidByOpponentModeling() {
		Bid bid = new Bid(negotiationSession.getDomain());
		try {
			HashMap<Integer, Value> valuesToOfferPerIssue = new HashMap<Integer, Value>();
			List<Issue> issues = negotiationSession.getIssues();

			// counters for each type of issue
			int discreteIndex = 0;
			int realIndex = 0;
			int integerIndex = 0;

			for (Issue lIssue : issues) {

				int issueNum = lIssue.getNumber();
				int indx = random100.nextInt(OPPONENT_ARRAY_SIZE);

				int first = 0;
				int last = 0;

				switch (lIssue.getType()) {

				case DISCRETE:
					HashMap<Value, Integer> valuesHash = new HashMap<Value, Integer>();
					valuesHash = opponentBidsStatisticsDiscrete.get(discreteIndex);

					// The keySet is the value that was proposed
					for (Value v : valuesHash.keySet()) {

						first = last;
						last = first + valuesHash.get(v);

						if (indx >= first && indx < last)
							valuesToOfferPerIssue.put(issueNum, v);
					}
					discreteIndex++;

					break;

				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					ArrayList<Integer> valueList = opponentBidsStatisticsForReal.get(realIndex);

					for (int i = 0; i < valueList.size(); i++) {

						first = last;
						last = first + valueList.get(i);

						if (indx >= first && indx <= last) {
							int lNrOfOptions = lIssueReal.getNumberOfDiscretizationSteps();
							double lOneStep = (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) / lNrOfOptions;
							double lowerBound = lIssueReal.getLowerBound();
							double realValueForBid = lowerBound + lOneStep * indx + random100.nextDouble() * lOneStep;
							ValueReal valueForBid = new ValueReal(realValueForBid);
							valuesToOfferPerIssue.put(issueNum, valueForBid);
						}
					}
					realIndex++;
					break;

				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					ArrayList<Integer> integerValueList = opponentBidsStatisticsForInteger.get(integerIndex);

					for (int i = 0; i < integerValueList.size(); i++) {
						first = last;
						last = first + integerValueList.get(i);

						if (indx >= first && indx <= last) {
							int valuesLowerBound = lIssueInteger.getLowerBound();
							ValueInteger valueIntegerForBid = new ValueInteger(valuesLowerBound + i);
							valuesToOfferPerIssue.put(issueNum, valueIntegerForBid);
						}
					}
					integerIndex++;
					break;
				}

				bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), valuesToOfferPerIssue);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return bid;
	}

	/**
	 * This function initializes the data structures that will be later used for
	 * the calculations of the statistics.
	 */
	private void initializeDataStructures() {
		try {
			opponentBidsStatisticsForReal = new ArrayList<ArrayList<Integer>>();
			opponentBidsStatisticsDiscrete = new ArrayList<HashMap<Value, Integer>>();
			opponentBidsStatisticsForInteger = new ArrayList<ArrayList<Integer>>();

			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

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
					int lNumOfPossibleValuesInThisIssue = lIssueReal.getNumberOfDiscretizationSteps();
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
					int lNumOfPossibleValuesForThisIssue = lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound()
							+ 1;
					for (int i = 0; i < lNumOfPossibleValuesForThisIssue; i++) {
						numOfValueProposals.add(0);
					}
					opponentBidsStatisticsForInteger.add(numOfValueProposals);
					break;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * fillBidsArray filling the array with random bids The maximum time that
	 * this function can run is TIME_TO_CREATE_BIDS_ARRAY seconds However, it
	 * will stop if all the possible bids were created
	 * 
	 * @param startTime
	 *            - the time when this function was called (in seconds, from the
	 *            beginning of the negotiation)
	 */
	private void fillBidsArray(double startTime) {
		int bidsMaxAmount = getBidMaxAmount();
		int countNewBids = 0;

		if (TEST_EQUIVALENCE) {
			BidIterator iterator = new BidIterator(negotiationSession.getUtilitySpace().getDomain());
			while (iterator.hasNext()) {
				ourBidsArray.add(iterator.next());
			}
		} else {
			while (new Date().getTime() - startTime < TIME_TO_CREATE_BIDS_ARRAY && countNewBids < bidsMaxAmount) {
				try {
					Bid newBid = getRandomBid();
					if (!ourBidsArray.contains(newBid)) {
						countNewBids++;
						ourBidsArray.add(newBid);
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * getBidMaxAmount counts how many possible bids exists in the given domain
	 * 
	 * @return the number of options
	 */
	private int getBidMaxAmount() {
		int count = 1;
		List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();
		for (Issue lIssue : issues) {
			switch (lIssue.getType()) {

			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				int numOfValues = lIssueDiscrete.getNumberOfValues();
				count = count * numOfValues;
				break;

			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				count = count * lIssueReal.getNumberOfDiscretizationSteps();
				break;

			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				// number of possible value when issue is integer (we should add
				// 1 in order to include all values)
				count = count * (lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound() + 1);
				break;
			}
		}
		return count;
	}

	/**
	 * @return a random bid
	 * @throws Exception
	 *             if we can't compute the utility (no evaluators have been set)
	 *             or when other evaluators than a DiscreteEvaluator are present
	 *             in the utility space.
	 */
	private Bid getRandomBid() {
		Bid bid = null;
		try {
			HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																			// <issuenumber,chosen
																			// value
																			// string>
			List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {

				case DISCRETE:

					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = random200.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
					break;

				case REAL:

					IssueReal lIssueReal = (IssueReal) lIssue;
					int lNrOfOptions = lIssueReal.getNumberOfDiscretizationSteps();
					double lOneStep = (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) / lNrOfOptions;
					int lOptionIndex = random200.nextInt(lNrOfOptions);
					if (lOptionIndex >= lNrOfOptions)
						lOptionIndex = lNrOfOptions - 1;
					ValueReal value = new ValueReal(
							lIssueReal.getLowerBound() + lOneStep * lOptionIndex + random200.nextDouble() * lOneStep);
					values.put(lIssueReal.getNumber(), value);
					break;

				case INTEGER:

					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					// number of possible value when issue is integer
					int numOfPossibleIntVals = lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound();
					int randomIndex = random200.nextInt(numOfPossibleIntVals) + lIssueInteger.getLowerBound();
					ValueInteger randomValueInteger = new ValueInteger(randomIndex);
					values.put(lIssue.getNumber(), randomValueInteger);
					break;
				}
			}
			bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		return bid;
	}

	/**
	 * selectCurrentBid - This function selects the next bid to offer to the
	 * opponent. The bid is selected randomly, with skips up and down.
	 * 
	 * @return
	 */
	private Bid selectCurrentBidFromOurBidsArray() {
		int value = random300.nextInt(randomInterval);
		int rndNum = value - randomOffset;
		int arraySize = ourBidsArray.size();
		int newIndex = 0;

		if (lastPositionInBidArray + rndNum < 0)// If the index is smaller than
												// the lower bound of the array
												// receiveMessage it to the
												// first cell
			newIndex = 0;
		else if (lastPositionInBidArray + rndNum > (arraySize - 1))// If the
																	// index is
																	// larger
																	// than the
																	// upper
																	// bound of
																	// the array
																	// receiveMessage
																	// it to the
																	// last cell
			newIndex = arraySize - 1;
		else
			newIndex = lastPositionInBidArray + rndNum;
		while ((bidsCountProposalArray[newIndex] / numOfProposalsFromOurBidsArray) > FREQUENCY_OF_PROPOSAL) {// If
																												// this
																												// bid
																												// was
																												// proposed
																												// too
																												// much
																												// than
																												// choose
																												// the
																												// next(neighbor)
																												// bid
			newIndex++;
		}
		Bid toSend = ourBidsArray.get(newIndex);
		// ADDED *********************************//
		if (this.negotiationSession.getUtilitySpace().getUtilityWithDiscount(toSend,
				negotiationSession.getTimeline()) < threshold) {
			toSend = previousOfferedBid;
			bidsCountProposalArray[lastPositionInBidArray]++;// receiveMessage
																// the number of
																// times that
																// this bid was
																// offered
		} else {
			previousOfferedBid = toSend;
			lastPositionInBidArray = newIndex;// receiveMessage the last
												// position - this is an
												// indication to the last bid
												// that was offered
			bidsCountProposalArray[newIndex]++;// receiveMessage the number of
												// times that this bid was
												// offered
		}

		return toSend;
	}

	/**
	 * initializeBidsFrequencyArray initializes all of the cells of the
	 * bidsCountProposalArray to 0
	 */
	private void initializeBidsFrequencyArray() {

		bidsCountProposalArray = new int[ourBidsArray.size()];
		for (int i = 0; i < bidsCountProposalArray.length; i++) {
			bidsCountProposalArray[i] = 0;
		}
	}

	@Override
	public String getName() {
		return "2011 - BRAMAgent";
	}
}
