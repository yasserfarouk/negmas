package agents.anac.y2012.MetaAgent.agents.MrFriendly;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;

public class MrFriendly extends Agent {

	/**
	 * When current time fraction is smaller than this boundary, we execute
	 * strategy 1. This means that if we believe our opponent stalls, we won't
	 * give in until we are in the last 10% of negotiations
	 */
	private static final double FIRST_TIME_BOUNDARY = 0.90;

	/**
	 * Minimum number of opponent bids we want before we switch from
	 * not-giving-in strategy to using the model to find a Pareto solution
	 */
	private static final double MIN_OPPONENT_BIDS_FOR_MODEL = 100;

	/**
	 * The (static) reservation value that we choose (no bids lower than this
	 * value are accepted)
	 */
	private static final double RESERVATION_VALUE = 0.3;

	/**
	 * The (dynamic) utility threshold above which we will always accept a bid.
	 * We keep updating this value and as time goes by, it should typically
	 * decrease slowly (but not during the first minute, where we try to model
	 * the opponent and don't give in much.. we start greedy)
	 */
	private double alwaysAcceptUtility;

	/**
	 * Keeps track of our opponent, estimates his issue weights and value
	 * preferences
	 */
	private OpponentModel opponentModel;

	/**
	 * BidTable object, used to get valuable information from bidding history.
	 * Also responsible for choosing our next offer
	 */
	private BidTable bidTable;

	/**
	 * Tells us wether or not we are negotiating on a domain where utility is
	 * discounted over time
	 */
	private boolean discountedDomain = false;

	/**
	 * Discount factor
	 */
	private double discountFactor;

	public void init() {
		double minimumBidUtility = RESERVATION_VALUE;
		try {
			alwaysAcceptUtility = getUtility(this.utilitySpace
					.getMaxUtilityBid());
			// If the utility of the highest bid possible is lower than the
			// reservation value, don't use
			// it and set the minimum bid utility to zero.
			if (alwaysAcceptUtility < minimumBidUtility) {
				minimumBidUtility = 0;
			}
		} catch (Exception e) {
			// If there are no possible bids in this domain, just set the always
			// accept utility to 1.
			alwaysAcceptUtility = 1;
		}
		discountFactor = this.utilitySpace.getDiscountFactor();
		discountedDomain = utilitySpace.isDiscounted();
		opponentModel = new OpponentModel(this.utilitySpace.getDomain()
				.getIssues(), discountFactor, timeline);
		bidTable = new BidTable(this, utilitySpace, minimumBidUtility,
				opponentModel);
	}

	public void ReceiveMessage(Action opponentAction) {
		// Give the action to the history tracker.
		bidTable.addOpponentAction(opponentAction);
	}

	@Override
	public Action chooseAction() {

		// TODO: at itexvscypress discount, top0.5% is only 2bids, maybe adjust
		// to small domains..

		Bid bid = null;
		Action action = null;
		Bid partnerBid = null;
		try {
			partnerBid = bidTable.getLastOpponentBid();
			if (partnerBid != null) {
				double offeredUtility = getUtility(partnerBid);
				// factor in discount factor on alwaysAcceptUtility
				double acceptNowUtil = (discountedDomain) ? alwaysAcceptUtility
						* Math.pow(discountFactor, timeline.getTime())
						: alwaysAcceptUtility;

				if (offeredUtility >= acceptNowUtil
						|| bidTable.weHaveOfferedThisBefore(partnerBid)) {
					// just accept if utility offered is higher than the bid we
					// offered that is lowest
					// or if we are offered a bid that we have offered before
					action = new Accept(getAgentID(), partnerBid);
					return action;
				}
			}

			// // If the time is < .991, then we proceed as usual.
			// if( timeline.getTime() < LAST_CHANCE_TIME_BOUNDARY){
			// if we're on a discounted domain, start giving in a bit more after
			// 0.5 of the time (AND before .991 of the time)..
			// actually we only change that IF we stall as well, we give in
			if (discountedDomain && timeline.getTime() > 0.5) {
				if (timeline.getTime() < FIRST_TIME_BOUNDARY
						&& (bidTable.getNumberOfOpponentBids() < MIN_OPPONENT_BIDS_FOR_MODEL || (opponentModel
								.isStalling(bidTable
										.getConsecutiveBidsDifferent()) && !this
								.weAreStalling()))) {
					// (we're in the first 90% of the time) AND (we don't have
					// enough bids for our model yet,
					// OR (opponent stalls AND we dont) )
					bid = bidTable.getBestBid();
				} else {
					bid = bidTable.getBestBidUsingModel();
					// Remove the bid from the list(s) so we won't offer the
					// same bid over and over again.
					bidTable.removeBid(bid);
				}

			} else {
				// after 150 received bids we have a pretty good estimation of
				// our opponents preference profile
				// and will use the model for selecting a bid. If our opponent
				// stalls (no 150 bids within 2 minutes,
				// or keeps sending the same bids) we wait for the last 10% of
				// time before we start using the model.
				if (timeline.getTime() < FIRST_TIME_BOUNDARY
						&& (bidTable.getNumberOfOpponentBids() < MIN_OPPONENT_BIDS_FOR_MODEL || opponentModel
								.isStalling(bidTable
										.getConsecutiveBidsDifferent()))) {
					// (we're in the first 90% of the time) AND (we don't have
					// enough bids for our model yet,
					// OR opponent stalls)
					bid = bidTable.getBestBid();
				} else {
					bid = bidTable.getBestBidUsingModel();
					// Remove the bid from the list(s) so we won't offer the
					// same bid over and over again.
					bidTable.removeBid(bid);
				}
			}
			// }else{ // We are in the last part of the session ( >= .991% of
			// the time).
			// // Get the best bid that the opponent did so far.
			// bid = bidTable.getBestOpponentBidSoFar();
			// System.out.println("[ANAC2011G6] OK, last chance (time: >="+LAST_CHANCE_TIME_BOUNDARY+"): offering "+bid+".");
			// }

			// Create a new offer to return.
			action = new Offer(getAgentID(), bid);

			// Give the bid to the history tracker through the bidTable.
			bidTable.addOwnBid(bid);

			// Update the always accept utility
			alwaysAcceptUtility = Math
					.max(alwaysAcceptUtility, getUtility(bid)); // receiveMessage
																// always_accept
			if (alwaysAcceptUtility < bidTable.getMinimumBidUtility()) { // Check
																			// if
																			// the
																			// always
																			// accept
																			// value
																			// isn't
																			// below
																			// the
																			// minimum
																			// bid
																			// utility.
				alwaysAcceptUtility = bidTable.getMinimumBidUtility();
			}

		} catch (Exception e) {
			System.out.println("Exception in chooseAction:" + e.getMessage());
			// If something goes wrong, we try to offer the best bid in the
			// domain.
			try {
				Bid bestBid = utilitySpace.getMaxUtilityBid();
				Action offer = new Offer(getAgentID(), bestBid);
				action = offer;
			} catch (Exception e2) {
				System.out
						.println("Exception in chooseAction while retrieving best bid of the domain: "
								+ e2.getMessage());
				// If we cannot even retrieve the best bid, let's accept.
				action = new Accept(getAgentID(), partnerBid);
			}
		}

		return action;
	}

	/**
	 * Returns the number of consecutive bids we have done that were non-unique
	 * 
	 * @return int
	 */
	private boolean weAreStalling() {
		return this.bidTable.weAreStalling();
	}

	@Override
	public String getVersion() {
		return "5.0";
	}

}
