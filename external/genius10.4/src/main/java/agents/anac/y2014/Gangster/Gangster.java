package agents.anac.y2014.Gangster;

import java.util.ArrayList;
import java.util.List;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.utility.NonlinearUtilitySpace;

//
// 
// 
// @author Dave de Jonge
//
//
public class Gangster extends Agent {

	double RESERVATION_VALUE = 0.0;
	int NUM_ISSUES = -1;
	double DISCOUNT_FACTOR;

	// PARAMETERS
	// concession
	int MINIMAL_MAX_DISTANCE = 5;
	int INITIAL_MAX_DISTANCE;
	private int DECREASE_THRESHOLD = 4; // if local search finds more than this
										// amount of altruistic bids, decrease
										// the max distance.
	int DECREASE_AMOUNT = 1; // the amount by which the max distance is
								// decreased each time this happens
	double DEADLINE_MARGIN = 0.01; // we aim to reach u_star at t_star, which
									// equals DISCOUNT_FACTOR - DEADLINE_MARGIN
	double INITIAL_TARGET_UTILITY = 0.995;

	// regression
	int NUM_INTERVALS = 100; // we take one sample per interval.
	double WINDOW_SIZE = 0.1; // we look only at a certain number of intervals
								// in the past. e.g. if num intervals is 100 and
								// window size is 0.1, then the window consists
								// of 10 intervals.
	int MiN_NUM_SAMPLES = 5; // the minimum numbers of samples we need before
								// doing regression.

	// storage
	final int MAX_CAPACITY = 2 * 1000; // maximum size to avoid decrease in
										// performance.
	final int MAX_SIZE_AFTER_CLEANING = 2 * 250; // 2 times the minimum size we
													// need to maintain good
													// results.
	final int EXPECTED_NUM_PROPOSALS = 180 * 100; // the number of proposals we
													// expect to receive during
													// the entire negotiation.

	// genetic algorithm
	final int INIT_GENERATION_SIZE = 120; // the size of the initial generation.
	final int NUM_SURVIVORS = 10; // the number of samples that survive from
									// each generation.
	int MAX_NUM_GENERATIONS = 10; // the maximum number of generations befor the
									// gen alg returns.
	int MIN_DISTANCE = 1; // the minimum distance between any pair of elements
							// in a survivor set.

	String name = "Gangster";
	String version = "1.0";

	List<Issue> issues;
	Bid latestReceivedBid;
	double latestReceivedUtility;
	int numProposalsReceived = 0;

	GenAlg genAlg;
	BidStorage bidStorage;

	ArrayList<BidDetails> globalSearchResults = null;
	int nextSelfishBidToPropose = 0;
	int maxDistance;

	private int numGoodResultsFromLocalSearch;
	private ExtendedBidDetails lastProposedByUs;

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	@Override
	public void init() {

		try {

			// 0. Make sure it works in linear cases ;)
			if (!(utilitySpace instanceof NonlinearUtilitySpace)) {
				return;
			}

			issues = utilitySpace.getDomain().getIssues();

			RESERVATION_VALUE = utilitySpace.getReservationValueUndiscounted();
			NUM_ISSUES = issues.size();
			DISCOUNT_FACTOR = utilitySpace.getDiscountFactor();

			// set the initial maxDistance;
			maxDistance = (int) Math.round(1.5 * NUM_ISSUES);
			numGoodResultsFromLocalSearch = 0;

			bidStorage = new BidStorage(MAX_CAPACITY, MAX_SIZE_AFTER_CLEANING,
					EXPECTED_NUM_PROPOSALS);

			if (utilitySpace instanceof NonlinearUtilitySpace) {
				genAlg = new GenAlg((NonlinearUtilitySpace) utilitySpace,
						INIT_GENERATION_SIZE, NUM_SURVIVORS,
						MAX_NUM_GENERATIONS, MIN_DISTANCE);

			}
		} catch (Exception e) {

		}
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {

		try {

			// update history
			if (opponentAction instanceof Offer) { // NOTE: an offer is a bid,
													// together with the id of
													// the agent that made the
													// offer.

				double time = timeline.getTime();

				// store the bid and its utility, this is necessary in case we
				// want to accept it.
				latestReceivedBid = ((Offer) opponentAction).getBid();
				latestReceivedUtility = utilitySpace
						.getUtility(latestReceivedBid);

				// 0. Make sure it works in linear cases ;)
				if (!(utilitySpace instanceof NonlinearUtilitySpace)) {
					return;
				}

				numProposalsReceived++;

				if (latestReceivedUtility > RESERVATION_VALUE) {
					BidDetails bd = new BidDetails(latestReceivedBid,
							latestReceivedUtility, time);
					bidStorage.addOpponentBid(bd);
				}

			}

		} catch (Exception e) {

			latestReceivedUtility = 0;
		}

	}

	@Override
	public Action chooseAction() {

		try {

			// 0. Make sure it works in linear cases ;)
			if (!(utilitySpace instanceof NonlinearUtilitySpace)) {

				if (latestReceivedUtility > 0.8) {
					return new Accept(getAgentID(), latestReceivedBid);
				}

				Bid returnBid = utilitySpace.getDomain().getRandomBid(null);
				return new Offer(getAgentID(), returnBid);
			}

			// 1. Get current time (value between 0 and 1, where 1 represents
			// the deadline).
			double time = timeline.getTime();

			// 2. Calculate the minimum utility we are going to demand this
			// turn.
			// get the best bid proposed so far by opponent, in the past time n
			// time units.
			double ourTargetUtility = getTargetUtility(time,
					DISCOUNT_FACTOR - DEADLINE_MARGIN);

			bidStorage.setTargetUtility(ourTargetUtility);

			if (latestReceivedUtility >= ourTargetUtility) {

				// ACCEPT BID
				return new Accept(getAgentID(), latestReceivedBid);

			}

			// 3. Get the maximum distance to the latest bid from the opponent.
			maxDistance = getMaxDistance(time, maxDistance,
					numGoodResultsFromLocalSearch);

			// 4. Do global search. Repeat it until we have found something
			// selfish enough.
			while (!bidStorage.weHaveSelfishEnoughBids()) {

				ArrayList<BidDetails> globalSearchResults = genAlg
						.globalSearch();

				bidStorage.addAll(globalSearchResults, false);

				// recalculate target utility.

				ourTargetUtility = getTargetUtility(timeline.getTime(),
						DISCOUNT_FACTOR - DEADLINE_MARGIN);

				bidStorage.setTargetUtility(ourTargetUtility);

			}

			// 5. Do local search.
			// initiate a genetic algorithm that searches within the allowed
			// space.
			// the returned values are always altruistic enough so we don't need
			// to repeat this.
			if (latestReceivedBid != null) {

				ArrayList<BidDetails> localSearchResults = genAlg
						.localSearch(latestReceivedBid, maxDistance);

				// count how many bids we have found that are both selfish
				// enough and altruistic enough. Note that we already know they
				// must be altruistic enough, so
				// We only need to test wether they are selfish enough.
				numGoodResultsFromLocalSearch = 0;
				for (BidDetails bd : localSearchResults) {
					if (bd.getMyUndiscountedUtil() > 0.9 * ourTargetUtility) {
						numGoodResultsFromLocalSearch++;
					}
				}

				bidStorage.addAll(localSearchResults, true);
			}

			// 6. Get the best bid of all possible bids in storage
			ExtendedBidDetails nextBid = bidStorage.getNext(ourTargetUtility,
					maxDistance);

			if (nextBid == null) {
				proposeOrAccept(time, lastProposedByUs, ourTargetUtility);
			}

			// 7. Propose it, or accept an earlier proposal from the opponent
			return proposeOrAccept(time, nextBid, ourTargetUtility);

		} catch (Exception e) {

			if (lastProposedByUs == null || lastProposedByUs.bidDetails == null
					|| lastProposedByUs.bidDetails.getBid() == null) {
				return new Offer(getAgentID(),
						utilitySpace.getDomain().getRandomBid(null));
			}

			return new Offer(getAgentID(),
					lastProposedByUs.bidDetails.getBid());

		}

	}

	private int getMaxDistance(double time, int previousValue,
			int numSelfishAndAltruisticBids) {

		if (numSelfishAndAltruisticBids > DECREASE_THRESHOLD
				&& previousValue > MINIMAL_MAX_DISTANCE) {
			return previousValue - DECREASE_AMOUNT;
		} else {
			return previousValue;
		}

	}

	/**
	 * Compares my bid with the latest received proposal from the opponent and
	 * converts the best one of the two in an Action, which is returned.
	 * 
	 * @param myExtendedBid
	 * @return
	 * @throws Exception
	 */
	Action proposeOrAccept(double time, ExtendedBidDetails myExtendedBid,
			double ourTargetUtility) throws Exception {

		BidDetails earlierProposal = bidStorage.getReproposableBid(time,
				ourTargetUtility, (1 - time) + 0.01);

		if (earlierProposal != null) {

			// repropose the bid
			bidStorage.addBidProposedByUs(earlierProposal.getBid());
			return new Offer(getAgentID(), earlierProposal.getBid());

		} else {

			this.lastProposedByUs = myExtendedBid;

			bidStorage.addBidProposedByUs(myExtendedBid.bidDetails.getBid());

			// remove the bid from the list of candidates to propose. This way
			// we keep the storage clean, cause we know we will never propose it
			// again anyway.
			bidStorage.removeBid(myExtendedBid);

			return new Offer(getAgentID(), myExtendedBid.bidDetails.getBid());
		}

	}

	double previousCalculationTime = 0; // the time at which we did the previous
										// calculation of the target utility.
	double previousTarget = INITIAL_TARGET_UTILITY; // the previous value of the
													// target utility

	double getTargetUtility(double time, double t_star) {

		double u_star = bidStorage.getBestOfferedUtility(time,
				(1 - time) + 0.01);

		if (time < t_star) {
			return getTargetUtility1(time, t_star, u_star);
		} else {
			return u_star;
		}

	}

	double getTargetUtility1(double time, double t_star, double u_star) {

		double high = 0.5 * (1 + u_star); // half-way between bestProposedSoFar
											// and 1.
		// double low = 0.5* (u_star + RESERVATION_VALUE); //half-way between
		// bestProposedSoFar and rv.
		double low = u_star;

		double finalTargetUtility = (high - low)
				* Math.pow((t_star - time), 0.3) + low;

		// calculate which part of the interval has passed since last
		// calculation
		double ratio = (t_star - time) / (t_star - previousCalculationTime);

		// target utility decreases linearly from the previous value to the
		// final value
		double currentTargetUtility = ratio
				* (previousTarget - finalTargetUtility) + finalTargetUtility;

		// System.out.println("Current target utility: " +
		// currentTargetUtility);

		// reset these two values for the next calculation.
		previousCalculationTime = time;
		previousTarget = currentTargetUtility;

		return currentTargetUtility;
	}

	@Override
	public String getName() {

		return name;
	}

	@Override
	public String getVersion() {
		return version;
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}

}
