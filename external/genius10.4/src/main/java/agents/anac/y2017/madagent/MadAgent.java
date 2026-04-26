package agents.anac.y2017.madagent;

import java.util.List;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;

public class MadAgent extends AbstractNegotiationParty {

	private static final Random RANDOM = new Random();
	private static final int MAXIMUM_NUMBER_OF_TRIALS = 2000;

	/*
	 * -------------------------------- RISK FUNCTION
	 * -------------------------------- f <- Round number to fake (Agent will
	 * fake in every f rounds) c <- Risk constant p <- Risk parameter Formula ->
	 * f = c / 2 ^ p We choose 5 as our parameter because we want our agent to
	 * be both aggressive and defensive
	 */
	private static final double RISK_CONSTANT = 100000;
	private static final double RISK_PARAMETER = 5; // Risk Parameter: 0, 1, 2,
													// ..., 8, 9, 10
	private static final int ROUND_NUMBER_TO_FAKE = (int) (RISK_CONSTANT
			/ Math.pow(2, RISK_PARAMETER));

	private SortedOutcomeSpace sortedOutcomeSpace = null;
	private Bid lastReceivedBid = null;
	private Bid bestReceivedBid = null;
	private Bid secondBestBid = null;
	private String negotiationType = null;
	private double negotiationLimit = 0;
	private double numberOfRoundsPassed = 0;
	private double timeToGetAlmostMad = 0;
	private double timeToGetMad = 0;
	private double threshold = 0.8;
	private double currentThreshold = 0.8;
	/* Variables for Opponent Modeling */
	private int opponentTurn = 0; // Value to keep track opponents
	private int myTurn = 0; // Random value to choose one of the opponents and
							// use its preferences while generating bid
	private int shiftBids[] = null; // Index for bidsPreferredByOpponents for
									// each opponent
	private OpponentModel[] opponentModels = null; // Opponent Model for each
													// opponent.
	private List<List<Bid>> bidsPreferredByOpponents = null; // bidsPrefferredByOpponents
																// for each
																// opponent

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		System.out.println("Discount Factor is "
				+ getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is "
				+ getUtilitySpace().getReservationValueUndiscounted());

		sortedOutcomeSpace = new SortedOutcomeSpace(utilitySpace);
		threshold *= 1.125;

		shiftBids = new int[3];
		opponentModels = new OpponentModel[3];
		bidsPreferredByOpponents = new ArrayList<List<Bid>>();

		// Filling the array with empty values to avoid 'Null Pointer Exception'
		for (int i = 0; i < 3; i++) {
			opponentModels[i] = new OpponentModel(utilitySpace, threshold);
			bidsPreferredByOpponents.add(null);
		}

		try {
			bestReceivedBid = utilitySpace.getMinUtilityBid();
		} catch (Exception e) {
			System.err.println("An exception thrown at init..");
		}

		negotiationType = info.getDeadline().getType().toString();
		negotiationLimit = info.getDeadline().getValue();

		/* This values will be used for adapting threshold */
		timeToGetMad = negotiationLimit * 0.8; // Agent gets mad in the last 20%
												// of the negotiation
		timeToGetAlmostMad = negotiationLimit * 0.5; // Agent gets almost mad in
														// the last 50% of the
														// negotiation

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD)
			throw new IllegalStateException("need standard persistent data");

		/* Agent calculates the second best bid */
		try {
			calculateSecondBestBid();
		} catch (Exception e) {
			System.err.println(
					"An exception thrown while calculating the second best bid..");
		}
	}

	private void calculateSecondBestBid() throws Exception {
		for (double u = utilitySpace
				.getUtility(utilitySpace.getMaxUtilityBid()); true; u -= 0.01) {
			secondBestBid = sortedOutcomeSpace.getBidNearUtility(u).getBid();

			if (utilitySpace.getUtility(secondBestBid) != utilitySpace
					.getUtility(utilitySpace.getMaxUtilityBid()))
				break;
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) { // ...
																// Opponent's
																// turn ...
		super.receiveMessage(sender, action);

		/*
		 * If the action is an Offer, get the last received bid and use it to
		 * form Opponent Model
		 */
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();

			opponentModels[2].offer(lastReceivedBid, numberOfRoundsPassed);
			opponentModels[opponentTurn++ % 2].offer(lastReceivedBid,
					numberOfRoundsPassed);
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) { // ...
																				// Your
																				// agent's
																				// turn
																				// ...
		numberOfRoundsPassed++;

		if (lastReceivedBid == null) { // You are the starter party, offer the
										// best possible bid
			return new Offer(getPartyId(), getBestBidPossible());
		} else { // You are not the starter party
			/* Determine the best received bid */
			if (utilitySpace.getUtility(lastReceivedBid) > utilitySpace
					.getUtility(bestReceivedBid))
				bestReceivedBid = lastReceivedBid;

			/*
			 * If utility of the last received bid is higher than the threshold,
			 * accept the offer.
			 */
			/* Else, offer a new bid. */
			if (utilitySpace.getUtility(lastReceivedBid) > currentThreshold)
				return new Accept(getPartyId(), lastReceivedBid);
			else
				return new Offer(getPartyId(), getBestBidPossible());
		}
	}

	private Bid getBestBidPossible() {
		try {
			double currentStatus = getCurrentStatus();

			if (currentStatus <= negotiationLimit * 0.05) { // First 5% of the
															// negotiation
				return secondBestBid;
			} else if ((int) numberOfRoundsPassed % ROUND_NUMBER_TO_FAKE <= 10
					&& currentStatus <= negotiationLimit * 0.9) {
				return getFakeBid();
			} else {
				myTurn = RANDOM.nextInt(3); // Generate random value among {0,
											// 1, 2}
				calculateCurrentThreshold(currentStatus);

				if (currentStatus > negotiationLimit * 0.99 && utilitySpace
						.getUtility(bestReceivedBid) >= currentThreshold)
					return bestReceivedBid;
				else
					return getNiceBid(currentStatus);
			}
		} catch (Exception e) {
			System.err.println("An exception thrown while generating bid..");
		}

		return generateRandomBid(); // This line will never be executed!!
	}

	/* Current status is the time/number of rounds passed */
	private double getCurrentStatus() {
		/* If the negotiation is time limited, use time as current status */
		if (negotiationType.equals("TIME"))
			return timeline.getTime() * timeline.getTotalTime();

		/*
		 * If the negotiation is round limited, use number of rounds as current
		 * status
		 */
		return numberOfRoundsPassed;
	}

	/*
	 * At first 90% of negotiation, agent generates a random bid to fake his
	 * opponent with certain frequency
	 */
	private Bid getFakeBid() {
		for (int trial = 1; trial <= MAXIMUM_NUMBER_OF_TRIALS; trial++) {
			Bid bid = generateRandomBid();

			/*
			 * The utility of the bid should be greater than 80% of the
			 * threshold
			 */
			if (utilitySpace.getUtility(bid) >= threshold * 0.8)
				return bid;
		}

		return generateRandomBid();
	}

	private void calculateCurrentThreshold(double currentStatus) {
		/* Threshold value is updated according to the agent's boulware level */
		threshold = opponentModels[myTurn].getNewThreshold();

		if (numberOfRoundsPassed % 10 == 0) {
			currentThreshold = threshold;
		} else if (currentStatus > timeToGetAlmostMad) {
			currentThreshold = threshold * 0.95;

			if (currentStatus > timeToGetMad)
				currentThreshold = threshold * 0.9;
		}
	}

	/* Get a nice bid using Opponent Model */
	private Bid getNiceBid(double currentStatus) throws Exception {
		Bid bid = generateRandomBid();

		/*
		 * Shift Bids is for shifting the index if the bid at the current index
		 * is not accepted by opponent
		 */
		if (currentStatus > timeToGetAlmostMad) {
			getBidsPreferredByOpponent();
			bid = (bidsPreferredByOpponents.get(myTurn) != null)
					? bidsPreferredByOpponents.get(myTurn)
							.get(shiftBids[myTurn]++ % bidsPreferredByOpponents
									.get(myTurn).size())
					: utilitySpace.getMaxUtilityBid();

			if (utilitySpace.getUtility(bid) >= currentThreshold)
				return bid;
			else
				shiftBids[myTurn] = 0;
		}

		for (int trial = 1; trial <= MAXIMUM_NUMBER_OF_TRIALS; trial++) {
			bid = generateRandomBid();

			if (utilitySpace.getUtility(bid) >= currentThreshold)
				return bid;
		}

		/*
		 * If it cannot generate a random bid higher than current threshold in
		 * the maximum number of trials, it returns max utility bid
		 */
		return utilitySpace.getMaxUtilityBid();
	}

	private void getBidsPreferredByOpponent() throws Exception {
		opponentModels[myTurn].computeMostPreferredBid();

		bidsPreferredByOpponents.set(myTurn, null);
		bidsPreferredByOpponents.set(myTurn,
				opponentModels[myTurn].getAcceptableBids());

		sortBids(bidsPreferredByOpponents.get(myTurn));

		/* If there is no element in the list, just add one */
		if (bidsPreferredByOpponents.get(myTurn).size() == 0)
			bidsPreferredByOpponents.get(myTurn)
					.add(utilitySpace.getMaxUtilityBid());
	}

	private void sortBids(List<Bid> bids) {
		bids.sort(new Comparator<Bid>() {
			@Override
			public int compare(Bid a, Bid b) {
				if (utilitySpace.getUtility(a) < utilitySpace.getUtility(b))
					return 1;
				if (utilitySpace.getUtility(a) > utilitySpace.getUtility(b))
					return -1;

				return 0;
			}
		});
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

	@Override
	public HashMap<String, String> negotiationEnded(Bid acceptedBid) {
		System.out.println("Negotiation has ended..");
		return null;
	}
}
