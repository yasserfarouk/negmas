package agents.anac.y2015.AgentNeo;

import java.util.ArrayList;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * This is your negotiation party.
 */
public class Groupn extends AbstractNegotiationParty {

	private double MaxUtility;
	private double utilitythreshold;
	private double discountFactor;
	private BidOptions bidOptions;
	private Action ActionOfOpponent = null;
	private OwnBids OwnBids;
	private Bid Maximum_Utility_Bid;// the bid with the maximum utility over the
									// utility space.
	private double reservationValue;
	private Object IDOfOpponent = null;
	private ArrayList<ArrayList<Bid>> bidsBetweenUtility;
	private double minimumUtilityThreshold;
	private double maximumOfBid;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		bidOptions = new BidOptions();
		this.bidOptions.initializeDataStructures(utilitySpace.getDomain());
		bidsBetweenUtility = new ArrayList<ArrayList<Bid>>();

		this.discountFactor = 1;
		if (utilitySpace.getDiscountFactor() <= 1D
				&& utilitySpace.getDiscountFactor() > 0D) {
			this.discountFactor = utilitySpace.getDiscountFactor();
		}
		OwnBids = new OwnBids();
		try {
			this.Maximum_Utility_Bid = utilitySpace.getMaxUtilityBid();
			this.utilitythreshold = utilitySpace
					.getUtility(Maximum_Utility_Bid); // initial
														// utility
														// threshold
		} catch (Exception e) {
			throw new RuntimeException("init failed:" + e, e);
		}
		this.MaxUtility = this.utilitythreshold;
		this.reservationValue = 0;
		if (utilitySpace.getReservationValue() > 0) {
			this.reservationValue = utilitySpace.getReservationValue();
		}
		// this.chooseUtilityThreshold();
		this.calculateBidsBetweenUtility();
		maximumOfBid = this.utilitySpace.getDomain().getNumberOfPossibleBids();

	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		Action action = null;
		try {
			Bid bid = null;

			if (OwnBids.numOfBidsProposed() == 0) {
				bid = this.Maximum_Utility_Bid;
				System.out.println("Valid Actions == null");
				action = new Offer(getPartyId(), bid);
			} else if (OwnBids.numOfBidsProposed() <= 7) {

				System.out.println("Number of Proposed bids < 7");
				bid = this.Maximum_Utility_Bid;
				System.out.println(bid);
				action = new Offer(getPartyId(), bid);

			} else {
				System.out.println("After 7 Bids");
				bid = AgentOffer();
				System.out.println(bid);

				if (ActionOfOpponent instanceof Offer) {

					System.out.println(((Offer) ActionOfOpponent).getBid()
							+ "Opponent's Offer");
					Boolean IsAccept = AcceptOffer(
							((Offer) ActionOfOpponent).getBid(), bid);
					System.out.println(IsAccept);
					Boolean IsTerminate = TerminateNegotiation(bid);
					System.out.println(IsTerminate);

					if (IsAccept && !IsTerminate) {
						action = new Accept(getPartyId(),
								((Offer) ActionOfOpponent).getBid());
						System.out.println("accept the offer");
					} else if (IsAccept && IsTerminate) {
						if (this.utilitySpace
								.getUtility(((Offer) ActionOfOpponent)
										.getBid()) > this.reservationValue) {
							action = new Accept(getPartyId(),
									((Offer) ActionOfOpponent).getBid());
							System.out.println("we accept the offer RANDOMLY");
						} else {
							bid = this.Maximum_Utility_Bid;
							action = new Offer(getPartyId(), bid);
						}
					} else {

						action = new Offer(getPartyId(), bid);
					}

				} else if (ActionOfOpponent instanceof Accept) {
					System.out.println("When there is an Accept");
					bid = AgentOffer();
					Bid Oppbid = BidOptions.getLastBid();

					Boolean IsAccept = AcceptOffer(Oppbid, bid);
					Boolean IsTerminate = TerminateNegotiation(Oppbid);

					if (IsAccept && !IsTerminate) { // True False
						action = new Accept(getPartyId(),
								((Offer) ActionOfOpponent).getBid());
						System.out.println("accept the offer");
					} else if (IsAccept && IsTerminate) { // True True
						if (this.utilitySpace
								.getUtility(((Offer) ActionOfOpponent)
										.getBid()) > this.reservationValue) {
							action = new Accept(getPartyId(),
									((Offer) ActionOfOpponent).getBid());
							System.out.println("we accept the offer RANDOMLY");
						} else { // False False
							bid = this.Maximum_Utility_Bid;
							action = new Offer(getPartyId(), bid);
						}
					}
					action = new Offer(getPartyId(), bid);

				}
			}

			this.OwnBids.addBid(bid, utilitySpace);
			System.out.println(bid);
			System.out.println(action);

		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			action = new EndNegotiation(getPartyId());
		}
		return action;

	}

	private boolean AcceptOffer(Bid opponentBid, Bid ownBid) {

		System.out.println("Accept Proposed Offer()");

		double currentUtility = 0; // opponent's proposed bid
		double nextRoundUtility = 0; // Agent's next bid
		double maximumThresholdUtility = 0; // maximum bid based on threshold
		// this.concedeToOpponent = false;

		try {
			currentUtility = this.utilitySpace.getUtility(opponentBid);
			maximumThresholdUtility = this.MaxUtility;
		} catch (Exception e) {
			System.out.println(e.getMessage() + "Exception in AcceptOffer #1");
		}
		try {
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
		} catch (Exception e) {
			System.out.println(e.getMessage() + "Exception in AcceptOffer #2");
		}

		if (currentUtility >= this.utilitythreshold
				|| currentUtility >= nextRoundUtility) {
			return true; // SCENARIO 1 Opponent's proposed utility surpasses
							// threshold or Agent's next bid
		} else {
			// SCENARIO 2
			// if the opponent's proposed utility with discount is larger than
			// the predicted
			// maximum utility with discount, then accept it. It is the most the
			// Agent could ever achieve.
			double predictMaxUtility = maximumThresholdUtility
					* this.discountFactor;
			System.out.println(predictMaxUtility);
			System.out.println("SCENARIO 2a");
			double bestMaxUtility = this.utilitySpace.getUtilityWithDiscount(
					bidOptions.getBestBidInHistory(), timeline);
			// What the Agent could possibly achieve given what has bids have
			// been proposed
			System.out.println(bestMaxUtility);
			System.out.println("SCENARIO 2b");
			if (bestMaxUtility > predictMaxUtility) {
				try {
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace.getUtility(opponentBid) >= utilitySpace
							.getUtility(bidOptions.getBestBidInHistory())
							- 0.1 /* 0.01 */) {
						return true;
					} else {

						return false;
					}
				} catch (Exception e) {
					System.out.println("exception in AcceptOffer #3");
					return true;
				}
				// in comparison with the threshold value which varies with time
			} else if (bestMaxUtility > this.utilitythreshold
					* Math.pow(this.discountFactor, timeline.getTime())) {
				try {
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace.getUtility(opponentBid) >= utilitySpace
							.getUtility(bidOptions.getBestBidInHistory())
							- 0.1 /* 0.01 */) {
						return true;
					} else {
						// this.concedeToOpponent = true;
						return false;
					}
				} catch (Exception e) {
					System.out.println("exception in AcceptOffer #4");
					return true;
				}
			} else {
				return false; // The proposed offer is entirely unacceptable.
			}
		}

	}

	private Bid AgentOffer() {

		int x = 0;
		Bid bid = null;
		System.out.println("AgentOffer()");
		int value = bidOptions.SimilartoOpponent(x);
		System.out.println("SimilartoOppo()");
		double concession1 = 0.0;
		double concession2 = 0.0;

		if (OwnBids.numOfBidsProposed() < 55) {
			concession1 = 0.15;
			concession2 = 0.02;
		} else {
			concession1 = 0.25;
			concession2 = 0.10;
		}

		List<Bid> candidateBids = this.getBidsInRange(
				this.MaxUtility - concession1, this.MaxUtility - concession2);

		System.out.println(value);
		switch (value) {

		case 1:
			bid = bidOptions.ChooseBid1(candidateBids,
					this.utilitySpace.getDomain());
			break;
		case 2:
			bid = bidOptions.ChooseBid2(candidateBids,
					this.utilitySpace.getDomain());
			break;
		case 3:
			bid = bidOptions.ChooseBid3(candidateBids,
					this.utilitySpace.getDomain());
			break;
		}
		return bid;
	}

	private void calculateBidsBetweenUtility() {
		BidIterator myBidIterator = new BidIterator(
				this.utilitySpace.getDomain());

		try {
			// double maximumUtility =
			// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			double maximumUtility = this.MaxUtility;
			double minUtility = this.minimumUtilityThreshold;
			int maximumRounds = (int) ((maximumUtility - minUtility) / 0.01);
			// initalization for each arraylist storing the bids between each
			// range
			for (int i = 0; i < maximumRounds; i++) {
				ArrayList<Bid> BidList = new ArrayList<Bid>();
				// BidList.add(this.bid_maximum_utility);
				this.bidsBetweenUtility.add(BidList);
			}
			this.bidsBetweenUtility.get(maximumRounds - 1)
					.add(this.Maximum_Utility_Bid);
			// note that here we may need to use some trick to reduce the
			// computation cost (to be checked later);
			// add those bids in each range into the corresponding arraylist
			// int limits = 0;
			if (this.maximumOfBid < 50000) {
				while (myBidIterator.hasNext()) {
					Bid b = myBidIterator.next();
					for (int i = 0; i < maximumRounds; i++) {
						if (utilitySpace.getUtility(b) <= (i + 1) * 0.01
								+ minUtility
								&& utilitySpace.getUtility(b) >= i * 0.01
										+ minUtility) {
							this.bidsBetweenUtility.get(i).add(b);
							break;
						}
					}
					// limits++;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in calculateBidsBetweenUtility()");
			e.printStackTrace();
		}
	}

	private List<Bid> getBidsInRange(double lowerBound, double upperBound) {
		List<Bid> bidsInRange = new ArrayList<Bid>();
		try {
			int range = (int) ((upperBound - this.minimumUtilityThreshold)
					/ 0.01);
			System.out.println(range);

			int initial = (int) ((lowerBound - this.minimumUtilityThreshold)
					/ 0.01);
			System.out.println(initial);

			for (int i = initial; i < range; i++) {
				bidsInRange.addAll(this.bidsBetweenUtility.get(i));
				// System.out.println(this.bidsBetweenUtility.get(i));
			}
			if (bidsInRange.isEmpty()) {
				bidsInRange.add(this.Maximum_Utility_Bid);
			}
		} catch (Exception e) {
			System.out.println("Exception in getBidsInRange");
			e.printStackTrace();
		}
		return bidsInRange;
	}

	private boolean TerminateNegotiation(Bid ownBid) {

		System.out.println("TerminateNegotiation()");

		double currentUtility = 0;
		double nextRoundUtility = 0;
		double maximumUtility = 0;
		// this.concedeToOpponent = false;
		try {
			currentUtility = this.reservationValue;
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
			maximumUtility = this.MaxUtility;
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "Exception in method TerminateNegotiation part 1");
		}

		if (currentUtility >= this.utilitythreshold
				|| currentUtility >= nextRoundUtility) {
			return true;
		} else {
			double predictMaxUtility = maximumUtility * this.discountFactor;
			double currentMaxUtility = this.utilitySpace
					.getReservationValueWithDiscount(timeline);
			if (currentMaxUtility > predictMaxUtility /*
														 * && timeline.getTime()
														 * > this
														 * .concedeToDiscountingFactor
														 */) {
				return true; // since the reservation value is going to be the
								// most the Agent will get
			} else {
				return false;
			}
		}
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		// Here you can listen to other parties' messages
		this.ActionOfOpponent = action;
		this.IDOfOpponent = sender;

		if (ActionOfOpponent instanceof Offer) { // if Agent is not the first to
													// act, update opponent's
													// offer first
			this.bidOptions.addOpponentBid(((Offer) ActionOfOpponent).getBid(),
					utilitySpace.getDomain(), this.utilitySpace, IDOfOpponent);

		}

	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
