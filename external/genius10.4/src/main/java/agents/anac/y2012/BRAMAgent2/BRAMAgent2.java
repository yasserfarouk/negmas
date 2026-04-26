package agents.anac.y2012.BRAMAgent2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

public class BRAMAgent2 extends Agent {

	private boolean EQUIVELENCE_TEST = true;

	/* FINAL VARIABLES */
	private final double TIME_TO_CREATE_BIDS_ARRAY = 2.0;// The time that we
															// allocate to
															// creating the bids
															// array
	private final double FREQUENCY_OF_PROPOSAL = 0.2;// If the frequency of the
														// proposal is larger
														// than this variable
														// than we won't propose
														// it
	// The threshold will be calculated as percentage of the required utility
	// depending of the elapsed time

	private final double THRESHOLD_PERC_FLEXIBILITY_1 = 0.07;
	private final double THRESHOLD_PERC_FLEXIBILITY_2 = 0.15;
	private final double THRESHOLD_PERC_FLEXIBILITY_3 = 0.3;
	private final double THRESHOLD_PERC_FLEXIBILITY_4 = 0.6;

	// The number of opponent's bids that we save in order to learn its
	// preferences
	private final int OPPONENT_ARRAY_SIZE = 10;
	/* MEMBERS */
	private Action actionOfPartner;// The action of the opponent
	private Bid bestBid;// The best bid that our agent offered
	private double maxUtility;// The maximum utility that our agent can get
	private ArrayList<Bid> ourBidsArray;// An Array that contains all the bids
										// that our agent can offer
	private ArrayList<Bid> opponentBidsArray;// An Array that contains the last
												// 100 bids that the opponent
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
	private double offeredUtility;// The utility of the current bid that the
									// opponent had offered
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

	@Override
	public void init() {
		actionOfPartner = null;
		ourBidsArray = new ArrayList<Bid>();
		bidsCountProposalArray = null;
		lastPositionInBidArray = 0;
		numOfProposalsFromOurBidsArray = 0;
		randomInterval = 8;
		randomOffset = 4;
		opponentBidsArray = new ArrayList<Bid>();
		initializeDataStructures();
		try {
			bestBid = this.utilitySpace.getMaxUtilityBid();
			maxUtility = this.utilitySpace.getUtilityWithDiscount(bestBid,
					timeline);
			ourBidsArray.add(bestBid);// The offer with the maximum utility will
										// be offered at the beginning
			threshold = maxUtility;
			previousOfferedBid = bestBid;

			if (EQUIVELENCE_TEST) {
				random100 = new Random(100);
				random200 = new Random(200);
				random300 = new Random(300);
			} else {
				random100 = new Random();
				random200 = new Random();
				random300 = new Random();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getName() {
		return "BRAMAgent 2";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		Bid bid2offer = new Bid(utilitySpace.getDomain());
		threshold = getNewThreshold();// Update the threshold according to the
										// discount factor
		try {
			// If we start the negotiation, we will offer the bid with
			// the maximum utility for us
			if (actionOfPartner == null) {
				bid2offer = this.utilitySpace.getMaxUtilityBid();
				action = new Offer(this.getAgentID(), bid2offer);
			} else if (actionOfPartner instanceof Offer) {
				offeredUtility = this.utilitySpace.getUtilityWithDiscount(
						((Offer) actionOfPartner).getBid(), timeline);

				if (offeredUtility >= threshold)// If the utility of the bid
												// that we received from the
												// opponent
												// is larger than the threshold
												// that we ready to accept,
												// we will accept the offer
					action = new Accept(this.getAgentID(),
							((Offer) actionOfPartner).getBid());
				else {
					Bid bidToRemove = null;
					Bid opponentBid = ((Offer) actionOfPartner).getBid();
					Bid bidToOffer = null;
					if (opponentBidsArray.size() < OPPONENT_ARRAY_SIZE) {// In
																			// this
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
						opponentBidsArray.add(opponentBid);
						updateStatistics(opponentBid, false);
						bidToOffer = bestBid;
					} else {
						// Remove the oldest bid and receiveMessage the
						// statistics
						bidToRemove = opponentBidsArray.get(0);
						updateStatistics(bidToRemove, true);
						opponentBidsArray.remove(0);
						// Add the new bid of the opponent and receiveMessage
						// the
						// statistics
						opponentBidsArray.add(opponentBid);
						updateStatistics(opponentBid, false);
						// Calculate the bid that the agent will offer
						// bidToOffer is null if we want to end the negotiation
						bidToOffer = getBidToOffer();
						// System.out.println("Original BidOffer: " +
						// bidToOffer);

					}

					if (/* ( bidToOffer != null ) && */(offeredUtility >= this.utilitySpace
							.getUtilityWithDiscount(bidToOffer, timeline))) {
						action = new Accept(this.getAgentID(),
								((Offer) actionOfPartner).getBid());
					} else if (/* (bidToOffer == null )|| */
					((offeredUtility < this.utilitySpace
							.getReservationValueWithDiscount(timeline))
							&& (timeline.getTime() > 177.0 / 180.0))
							&& (this.utilitySpace
									.getReservationValueWithDiscount(
											timeline) > this.utilitySpace
													.getUtilityWithDiscount(
															bidToOffer,
															timeline))) {
						action = new EndNegotiation(this.getAgentID());
					} else {
						action = new Offer(this.getAgentID(), bidToOffer);
					}
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			action = new Accept(this.getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
			if (actionOfPartner != null)
				System.out.println(
						"BRAMAgent accepted the offer beacuse of an exception");
		}
		return action;
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
			List<Issue> issues = utilitySpace.getDomain().getIssues();

			// counters for each type of issue
			int realIndex = 0;
			int discreteIndex = 0;
			int integerIndex = 0;

			for (Issue lIssue : issues) {
				int issueNum = lIssue.getNumber();
				Value v = bidToUpdate.getValue(issueNum);
				switch (lIssue.getType()) {
				case DISCRETE:
					if (opponentBidsStatisticsDiscrete == null)
						System.out.println(
								"opponentBidsStatisticsDiscrete is NULL");
					else if (opponentBidsStatisticsDiscrete
							.get(discreteIndex) != null) {
						int counterPerValue = opponentBidsStatisticsDiscrete
								.get(discreteIndex).get(v);
						if (toRemove)
							counterPerValue--;
						else
							counterPerValue++;
						opponentBidsStatisticsDiscrete.get(discreteIndex).put(v,
								counterPerValue);
					}
					discreteIndex++;
					break;

				case REAL:

					IssueReal lIssueReal = (IssueReal) lIssue;
					int lNumOfPossibleRealValues = lIssueReal
							.getNumberOfDiscretizationSteps();
					double lOneStep = (lIssueReal.getUpperBound()
							- lIssueReal.getLowerBound())
							/ lNumOfPossibleRealValues;
					double first = lIssueReal.getLowerBound();
					double last = lIssueReal.getLowerBound() + lOneStep;
					double valueReal = ((ValueReal) v).getValue();
					boolean found = false;

					for (int i = 0; !found && i < opponentBidsStatisticsForReal
							.get(realIndex).size(); i++) {
						if (valueReal >= first && valueReal <= last) {
							int countPerValue = opponentBidsStatisticsForReal
									.get(realIndex).get(i);
							if (toRemove)
								countPerValue--;
							else
								countPerValue++;

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
						int countPerValue = opponentBidsStatisticsForReal
								.get(realIndex).get(i);
						if (toRemove)
							countPerValue--;
						else
							countPerValue++;

						opponentBidsStatisticsForReal.get(realIndex).set(i,
								countPerValue);
					}
					realIndex++;
					break;

				case INTEGER:

					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int valueInteger = ((ValueInteger) v).getValue();

					int valueIndex = valueInteger
							- lIssueInteger.getLowerBound(); // For
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
					int countPerValue = opponentBidsStatisticsForInteger
							.get(integerIndex).get(valueIndex);
					if (toRemove)
						countPerValue--;
					else
						countPerValue++;

					opponentBidsStatisticsForInteger.get(integerIndex)
							.set(valueIndex, countPerValue);
					integerIndex++;
					break;
				}
			}
		} catch (Exception ex) {
			System.out.println(
					"BRAM - Exception in updateStatistics: " + toRemove);
		}

	}

	/**
	 * This function calculates the threshold. It takes into consideration the
	 * time that passed from the beginning of the game. As time goes by, the
	 * agent becoming more flexible to the offers that it is willing to accept.
	 * 
	 * @return - the threshold
	 */
	private double getNewThreshold() {
		double minUtil = utilitySpace.getUtilityWithDiscount(
				ourBidsArray.get(ourBidsArray.size() - 1), timeline);
		double maxUtil = utilitySpace.getUtilityWithDiscount(bestBid, timeline);
		double tresholdBestBidDiscount = 0.0;

		if (timeline.getTime() < 60.0 / 180.0)
			tresholdBestBidDiscount = maxUtil
					- (maxUtil - minUtil) * THRESHOLD_PERC_FLEXIBILITY_1;
		else if (timeline.getTime() < 150.0 / 180.0)
			tresholdBestBidDiscount = maxUtil
					- (maxUtil - minUtil) * THRESHOLD_PERC_FLEXIBILITY_2;
		else if (timeline.getTime() < 175.0 / 180.0)
			tresholdBestBidDiscount = maxUtil
					- (maxUtil - minUtil) * THRESHOLD_PERC_FLEXIBILITY_3;
		else
			tresholdBestBidDiscount = maxUtil
					- (maxUtil - minUtil) * THRESHOLD_PERC_FLEXIBILITY_4;

		return tresholdBestBidDiscount;

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
			// System.out.println("Original Threshold: " + maxUt);

			// if (maxUt > endNegotiationUtility){
			for (int i = 0; i < 10; i++) {
				Bid currBid = createBidByOpponentModeling();
				if (currBid == null) {
					System.out.println(
							" BRAM - currBid in getBidToOffer is NULL");
				} else {
					// System.out.println(" BRAM - currBid: " +
					// currBid.toString());
					double currUtility = utilitySpace
							.getUtilityWithDiscount(currBid, timeline);

					if (currUtility > maxUt) {
						maxUt = currUtility;
						bidWithMaxUtility = currBid;
					}
				}

			}
			if (bidWithMaxUtility == null) {
				if (EQUIVELENCE_TEST) {
					return bestBid;
				} else {
					return getBidFromBidsArray();
				}
			} else {
				// System.out.println("****************BRAM opponent modeling");
				return bidWithMaxUtility;
			}
			// }
			// else{
			// return null;//The agent can't make a good enough bid and returns
			// null to end the negotiation - won't get here
			// }
		} catch (Exception e) {
			System.out.println("BRAM - Exception in GetBidToOffer function");
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
				@Override
				public int compare(Bid bid1, Bid bid2) {
					// We will sort the array in a descending order
					double utility1 = 0.0;
					double utility2 = 0.0;
					try {
						utility1 = utilitySpace.getUtility(bid1);
						utility2 = utilitySpace.getUtility(bid2);
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
		Bid bid = new Bid(utilitySpace.getDomain());
		try {

			HashMap<Integer, Value> valuesToOfferPerIssue = new HashMap<Integer, Value>();
			List<Issue> issues = utilitySpace.getDomain().getIssues();

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
					if (opponentBidsStatisticsDiscrete == null)
						System.out.println(
								"BRAM - opponentBidsStatisticsDiscrete IS NULL");
					valuesHash = opponentBidsStatisticsDiscrete
							.get(discreteIndex);

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
					ArrayList<Integer> valueList = opponentBidsStatisticsForReal
							.get(realIndex);

					for (int i = 0; i < valueList.size(); i++) {

						first = last;
						last = first + valueList.get(i);

						if (indx >= first && indx <= last) {
							int lNrOfOptions = lIssueReal
									.getNumberOfDiscretizationSteps();
							double lOneStep = (lIssueReal.getUpperBound()
									- lIssueReal.getLowerBound())
									/ lNrOfOptions;
							double lowerBound = lIssueReal.getLowerBound();
							double realValueForBid = lowerBound
									+ lOneStep * indx
									+ random100.nextDouble() * lOneStep;
							ValueReal valueForBid = new ValueReal(
									realValueForBid);
							valuesToOfferPerIssue.put(issueNum, valueForBid);
						}
					}
					realIndex++;
					break;

				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					ArrayList<Integer> integerValueList = opponentBidsStatisticsForInteger
							.get(integerIndex);

					for (int i = 0; i < integerValueList.size(); i++) {
						first = last;
						last = first + integerValueList.get(i);

						if (indx >= first && indx <= last) {
							int valuesLowerBound = lIssueInteger
									.getLowerBound();
							ValueInteger valueIntegerForBid = new ValueInteger(
									valuesLowerBound + i);
							valuesToOfferPerIssue.put(issueNum,
									valueIntegerForBid);
						}
					}
					integerIndex++;
					break;
				}

				bid = new Bid(utilitySpace.getDomain(), valuesToOfferPerIssue);
			}
		} catch (Exception e) {
			System.out.println(
					"BRAM - Exception in createBidByOpponentModeling function");
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

			List<Issue> issues = utilitySpace.getDomain().getIssues();

			for (Issue lIssue : issues) {

				switch (lIssue.getType()) {

				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					HashMap<Value, Integer> discreteIssueValuesMap = new HashMap<Value, Integer>();
					for (int j = 0; j < lIssueDiscrete
							.getNumberOfValues(); j++) {
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
							.getUpperBound() - lIssueInteger.getLowerBound()
							+ 1;
					for (int i = 0; i < lNumOfPossibleValuesForThisIssue; i++) {
						numOfValueProposals.add(0);
					}
					opponentBidsStatisticsForInteger.add(numOfValueProposals);
					break;
				}
			}
		} catch (Exception e) {
			System.out.println("BRAM - EXCEPTION in initializeDataAtructures");
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
		while (new Date().getTime() - startTime < TIME_TO_CREATE_BIDS_ARRAY
				&& countNewBids < bidsMaxAmount) {
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

	/**
	 * getBidMaxAmount counts how many possible bids exists in the given domain
	 * 
	 * @return the number of options
	 */
	private int getBidMaxAmount() {
		int count = 1;
		List<Issue> issues = utilitySpace.getDomain().getIssues();
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
				count = count * (lIssueInteger.getUpperBound()
						- lIssueInteger.getLowerBound() + 1);
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
			List<Issue> issues = utilitySpace.getDomain().getIssues();

			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {

				case DISCRETE:

					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = random200
							.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;

				case REAL:

					IssueReal lIssueReal = (IssueReal) lIssue;
					int lNrOfOptions = lIssueReal
							.getNumberOfDiscretizationSteps();
					double lOneStep = (lIssueReal.getUpperBound()
							- lIssueReal.getLowerBound()) / lNrOfOptions;
					int lOptionIndex = random200.nextInt(lNrOfOptions);
					if (lOptionIndex >= lNrOfOptions)
						lOptionIndex = lNrOfOptions - 1;
					ValueReal value = new ValueReal(
							lIssueReal.getLowerBound() + lOneStep * lOptionIndex
									+ random200.nextDouble() * lOneStep);
					values.put(lIssueReal.getNumber(), value);
					break;

				case INTEGER:

					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					// number of possible value when issue is integer
					int numOfPossibleIntVals = lIssueInteger.getUpperBound()
							- lIssueInteger.getLowerBound();
					int randomIndex = random200.nextInt(numOfPossibleIntVals)
							+ lIssueInteger.getLowerBound();
					ValueInteger randomValueInteger = new ValueInteger(
							randomIndex);
					values.put(lIssue.getNumber(), randomValueInteger);
					break;
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} catch (Exception ex) {
			System.out.println("BRAM - Exception in getRandomBid");
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
		int rndNum = random300.nextInt(randomInterval) - randomOffset;
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
																	// it
																	// to the
																	// last cell
			newIndex = arraySize - 1;
		else
			newIndex = lastPositionInBidArray + rndNum;
		while ((bidsCountProposalArray[newIndex]
				/ numOfProposalsFromOurBidsArray) > FREQUENCY_OF_PROPOSAL) {// If
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
		if (this.utilitySpace.getUtilityWithDiscount(toSend,
				timeline) < threshold) {
			toSend = previousOfferedBid;
			bidsCountProposalArray[lastPositionInBidArray]++;// receiveMessage
																// the
																// number of
																// times that
																// this bid was
																// offered
		} else {
			previousOfferedBid = toSend;
			lastPositionInBidArray = newIndex;// receiveMessage the last
												// position - this
												// is an indication to the last
												// bid that was offered
			bidsCountProposalArray[newIndex]++;// receiveMessage the number of
												// times
												// that this bid was offered
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
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2012";
	}
}
