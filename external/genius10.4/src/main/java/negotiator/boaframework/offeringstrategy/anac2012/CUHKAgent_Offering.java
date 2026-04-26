package negotiator.boaframework.offeringstrategy.anac2012;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import agents.anac.y2012.CUHKAgent.OwnBidHistory;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.offeringstrategy.anac2012.CUHKAgent.OpponentBidHistory;
import negotiator.boaframework.opponentmodel.DefaultModel;

/**
 * This is the decoupled Bidding Strategy of CUHKAgent Note that the Opponent
 * Model was not decoupled and thus is integrated into this strategy As well due
 * to the strong coupling with the AC there is not AC_CUHKAgent and thus it is
 * not proven equivalent with the original This is the offering and does not
 * simulate the stopping situation of CUHKAgent
 * 
 * The AC and reservation value code was removed
 * 
 * @author Alex Dirkzwager
 */

public class CUHKAgent_Offering extends OfferingStrategy {

	private double totalTime;
	private double maximumOfBid;
	private OwnBidHistory ownBidHistory;
	private OpponentBidHistory opponentBidHistory;
	private double minimumUtilityThreshold;
	private double utilitythreshold;
	private double MaximumUtility;
	private double timeLeftBefore;
	private double timeLeftAfter;
	private double maximumTimeOfOpponent;
	private double maximumTimeOfOwn;
	private double discountingFactor;
	private double concedeToDiscountingFactor;
	private double concedeToDiscountingFactor_original;
	private double minConcedeToDiscountingFactor;
	private ArrayList<ArrayList<Bid>> bidsBetweenUtility;
	private boolean concedeToOpponent;
	private boolean toughAgent; // if we propose a bid that was proposed by the
								// opponnet, then it should be accepted.
	private double alpha1;// the larger alpha is, the more tough the agent is.
	private Bid bid_maximum_utility;// the bid with the maximum utility over the
									// utility space.
	private double reservationValue;
	private Random random;

	public CUHKAgent_Offering() {
	}

	public CUHKAgent_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) throws Exception {
		init(negoSession, model, oms, null);
	}

	/**
	 * Init required for the Decoupled Framework.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		random = new Random();
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		this.opponentModel = model;
		negotiationSession = negoSession;
		maximumOfBid = negoSession.getUtilitySpace().getDomain().getNumberOfPossibleBids();
		ownBidHistory = new OwnBidHistory();
		opponentBidHistory = new OpponentBidHistory(opponentModel, oms,
				(AdditiveUtilitySpace) negotiationSession.getUtilitySpace());
		bidsBetweenUtility = new ArrayList<ArrayList<Bid>>();

		this.bid_maximum_utility = negoSession.getUtilitySpace().getMaxUtilityBid();
		this.utilitythreshold = negoSession.getUtilitySpace().getUtility(bid_maximum_utility); // initial
																								// utility
																								// threshold
		this.MaximumUtility = this.utilitythreshold;
		this.timeLeftAfter = 0;
		this.timeLeftBefore = 0;
		this.totalTime = negoSession.getTimeline().getTotalTime();
		this.maximumTimeOfOpponent = 0;
		this.maximumTimeOfOwn = 0;
		this.minConcedeToDiscountingFactor = 0.08;// 0.1;
		this.discountingFactor = 1;
		if (negoSession.getUtilitySpace().getDiscountFactor() <= 1D
				&& negoSession.getUtilitySpace().getDiscountFactor() > 0D) {
			this.discountingFactor = negoSession.getUtilitySpace().getDiscountFactor();
		}
		this.chooseUtilityThreshold();
		this.calculateBidsBetweenUtility();
		this.chooseConcedeToDiscountingDegree();
		this.opponentBidHistory.initializeDataStructures(negoSession.getUtilitySpace().getDomain());
		this.timeLeftAfter = negoSession.getTimeline().getCurrentTime();
		this.concedeToOpponent = false;
		this.toughAgent = false;
		this.alpha1 = 2;
		this.reservationValue = 0;
		if (negoSession.getUtilitySpace().getReservationValue() > 0) {
			this.reservationValue = negoSession.getUtilitySpace().getReservationValue();
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public BidDetails determineNextBid() {
		BidDetails action = null;
		try {

			this.timeLeftBefore = negotiationSession.getTimeline().getCurrentTime();
			BidDetails bid = null;
			// we propose first and propose the bid with maximum utility
			if (negotiationSession.getOpponentBidHistory().isEmpty()) {
				bid = negotiationSession.getMaxBidinDomain();
				action = bid;
			} else {
				// receiveMessage opponent model first
				this.opponentBidHistory.updateOpponentModel(negotiationSession.getOpponentBidHistory().getLastBid(),
						negotiationSession.getUtilitySpace().getDomain(),
						(AdditiveUtilitySpace) negotiationSession.getUtilitySpace());
				this.updateConcedeDegree();
				// receiveMessage the estimation
				if (negotiationSession.getOwnBidHistory().size() == 0) {
					// bid = utilitySpace.getMaxUtilityBid();
					bid = negotiationSession.getMaxBidinDomain();
					action = bid;
				} else {// other conditions
					if (estimateRoundLeft(true) > 10) {// still have some rounds
														// left to further
														// negotiate (the major
														// negotiation period)
						bid = BidToOffer();
						/** REMOVED AC */
						// Boolean IsAccept =
						// AcceptOpponentOffer(negotiationSession.getOpponentBidHistory().getLastBid(),
						// bid.getBid());
						// Boolean IsTerminate =
						// TerminateCurrentNegotiation(bid.getBid());

						// if(!IsAccept && !IsTerminate){
						// we expect that the negotiation is over once we select
						// a bid from the opponent's history.
						if (this.concedeToOpponent == true) {
							// bid =
							// opponentBidHistory.chooseBestFromHistory(this.utilitySpace);
							Bid possibleBid = opponentBidHistory.getBestBidInHistory();
							action = new BidDetails(possibleBid,
									negotiationSession.getUtilitySpace().getUtility(possibleBid));
							this.toughAgent = true;
							this.concedeToOpponent = false;
						} else {
							action = bid;
							this.toughAgent = false;
						}
						// }
					} else {// this is the last chance and we concede by
							// providing the opponent the best offer he ever
							// proposed to us
							// in this case, it corresponds to an opponent whose
							// decision time is short
						if (negotiationSession.getTimeline().getTime() > 0.9985 && estimateRoundLeft(true) < 5) {
							// bid =
							// opponentBidHistory.chooseBestFromHistory(this.utilitySpace);
							Bid bid1 = opponentBidHistory.getBestBidInHistory();
							bid = new BidDetails(bid1, negotiationSession.getUtilitySpace().getUtility(bid1));

							// this is specially designed to avoid that we got
							// very low utility by searching between an
							// acceptable range (when the domain is small)
							if (negotiationSession.getUtilitySpace().getUtility(bid.getBid()) < 0.85) {
								List<Bid> candidateBids = this.getBidsBetweenUtility(this.MaximumUtility - 0.15,
										this.MaximumUtility - 0.02);
								// if the candidate bids do not exsit and also
								// the deadline is approaching in next round, we
								// concede.
								// if (candidateBids.size() == 1 &&
								// timeline.getTime()>0.9998) {
								// we have no chance to make a new proposal
								// before the deadline
								if (this.estimateRoundLeft(true) < 2) {
									Bid bid2 = opponentBidHistory.getBestBidInHistory();
									bid = new BidDetails(bid2, negotiationSession.getUtilitySpace().getUtility(bid2));
								} else {
									Bid possibleBid = opponentBidHistory.ChooseBid(candidateBids,
											negotiationSession.getUtilitySpace().getDomain());
									bid = new BidDetails(possibleBid,
											negotiationSession.getUtilitySpace().getUtility(possibleBid));
								}
							}

							/**
							 * REMOVED AC Boolean IsAccept =
							 * AcceptOpponentOffer(
							 * negotiationSession.getOpponentBidHistory
							 * ().getLastBid(), bid.getBid()); Boolean
							 * IsTerminate =
							 * TerminateCurrentNegotiation(bid.getBid());
							 * if(!IsAccept && !IsTerminate){ if
							 * (this.toughAgent == false) { action = bid;
							 * //this.toughAgent = true; } }
							 */
							action = bid;
							// in this case, it corresponds to the situation
							// that we encounter an opponent who needs more
							// computation to make decision each round
						} else {// we still have some time to negotiate,
							// and be tough by sticking with the lowest one in
							// previous offer history.
							// we also have to make the decisin fast to avoid
							// reaching the deadline before the decision is made
							// bid = ownBidHistory.GetMinBidInHistory();//reduce
							// the computational cost
							bid = BidToOffer();

							/** REMOVED AC */
							// Boolean IsAccept =
							// AcceptOpponentOffer(negotiationSession.getOpponentBidHistory().getLastBid(),bid.getBid());
							// Boolean IsTerminate =
							// TerminateCurrentNegotiation(bid.getBid());
							// if(!IsAccept && !IsTerminate){
							action = bid;
							// }
						}
					}
				}
			}

			this.ownBidHistory.addBid(bid.getBid(), (AdditiveUtilitySpace) negotiationSession.getUtilitySpace());
			this.timeLeftAfter = negotiationSession.getTimeline().getCurrentTime();
			this.estimateRoundLeft(false);// receiveMessage the estimation
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			System.out.println(estimateRoundLeft(false));
		}

		return action;
	}

	/*
	 * principle: randomization over those candidate bids to let the opponent
	 * have a better model of my utility profile return the bid to be offered in
	 * the next round
	 */
	private BidDetails BidToOffer() {
		Bid bidReturned = null;
		double decreasingAmount_1 = 0.05;
		double decreasingAmount_2 = 0.25;
		try {
			double maximumOfBid = this.MaximumUtility;// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			double minimumOfBid;
			// used when the domain is very large.
			// make concession when the domin is large
			if (this.discountingFactor == 1 && this.maximumOfBid > 3000) {
				minimumOfBid = this.MaximumUtility - decreasingAmount_1;
				// make further concession when the deadline is approaching and
				// the domain is large
				if (this.discountingFactor > 1 - decreasingAmount_2 && this.maximumOfBid > 10000
						&& negotiationSession.getTimeline().getTime() >= 0.98) {
					minimumOfBid = this.MaximumUtility - decreasingAmount_2;
				}
				if (this.utilitythreshold > minimumOfBid) {
					this.utilitythreshold = minimumOfBid;
				}
			} /*
				 * else if (this.discountingFactor > 1 - decreasingAmount_3 &&
				 * this.maximumOfBid >= 100000 && this.maximumOfBid < 300000) {
				 * minimumOfBid = this.MaximumUtility - decreasingAmount_3; }
				 * else if (this.discountingFactor > 1 - decreasingAmount_4 &&
				 * this.maximumOfBid >= 300000) { minimumOfBid =
				 * this.MaximumUtility - decreasingAmount_4; }
				 */else {// the general case
				if (negotiationSession.getTimeline().getTime() <= this.concedeToDiscountingFactor) {
					double minThreshold = (maximumOfBid * this.discountingFactor)
							/ Math.pow(this.discountingFactor, this.concedeToDiscountingFactor);
					this.utilitythreshold = maximumOfBid - (maximumOfBid - minThreshold) * Math.pow(
							(negotiationSession.getTimeline().getTime() / this.concedeToDiscountingFactor), alpha1);
				} else {
					this.utilitythreshold = (maximumOfBid * this.discountingFactor)
							/ Math.pow(this.discountingFactor, negotiationSession.getTimeline().getTime());
				}
				minimumOfBid = this.utilitythreshold;
			}

			/*
			 * if(minimumOfBid < 0.9 && this.guessOpponentType == false){
			 * if(this.opponentBidHistory.getSize() <= 2){ this.opponentType =
			 * 1;//tough opponent alpha1 = 2; } else{ this.opponentType = 0;
			 * alpha1 = 4; } this.guessOpponentType = true;//we only guess the
			 * opponent type once here System.out.println("we guess the opponent
			 * type is "+this.opponentType); }
			 */
			// choose from the opponent bid history first to reduce calculation
			// time
			Bid bestBidOfferedByOpponent = opponentBidHistory.getBestBidInHistory();
			if (negotiationSession.getUtilitySpace().getUtility(bestBidOfferedByOpponent) >= this.utilitythreshold
					|| negotiationSession.getUtilitySpace().getUtility(bestBidOfferedByOpponent) >= minimumOfBid) {
				return new BidDetails(bestBidOfferedByOpponent,
						negotiationSession.getUtilitySpace().getUtility(bestBidOfferedByOpponent));
			}

			List<Bid> candidateBids = this.getBidsBetweenUtility(minimumOfBid, maximumOfBid);
			bidReturned = opponentBidHistory.ChooseBid(candidateBids,
					this.negotiationSession.getUtilitySpace().getDomain());
			if (bidReturned == null) {
				System.out.println("no bid is searched warning");
				bidReturned = this.negotiationSession.getUtilitySpace().getMaxUtilityBid();
			}
		} catch (Exception e) {
			System.out.println(e.getMessage() + "exception in method BidToOffer");
		}
		try {
			return new BidDetails(bidReturned, negotiationSession.getUtilitySpace().getUtility(bidReturned));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * REMOVED AC private boolean AcceptOpponentOffer(Bid opponentBid, Bid
	 * ownBid) { double currentUtility = 0; double nextRoundUtility = 0; double
	 * maximumUtility = 0; this.concedeToOpponent = false; try { currentUtility
	 * = this.negotiationSession.getUtilitySpace().getUtility(opponentBid);
	 * maximumUtility = this.MaximumUtility; } catch (Exception e) {
	 * System.out.println(e.getMessage() + "Exception in method
	 * AcceptOpponentOffer part 1"); } try { nextRoundUtility =
	 * this.negotiationSession.getUtilitySpace().getUtility(ownBid); } catch
	 * (Exception e) { System.out.println(e.getMessage() + "Exception in method
	 * AcceptOpponentOffer part 2"); }
	 * //System.out.println(this.utilitythreshold +"at time "+
	 * timeline.getTime()); if (currentUtility >= this.utilitythreshold ||
	 * currentUtility >= nextRoundUtility) { return true; } else { //if the
	 * current utility with discount is larger than the predicted maximum
	 * utility with discount //then accept it. double predictMaximumUtility =
	 * maximumUtility * this.discountingFactor; //double currentMaximumUtility =
	 * this.utilitySpace.getUtilityWithDiscount(opponentBidHistory.
	 * chooseBestFromHistory(utilitySpace), timeline); double
	 * currentMaximumUtility =
	 * this.negotiationSession.getUtilitySpace().getUtilityWithDiscount
	 * (opponentBidHistory.getBestBidInHistory(),
	 * negotiationSession.getTimeline()); if (currentMaximumUtility >
	 * predictMaximumUtility && negotiationSession.getTimeline().getTime() >
	 * this.concedeToDiscountingFactor) { try { //if the current offer is
	 * approximately as good as the best one in the history, then accept it. if
	 * (negotiationSession.getUtilitySpace().getUtility(opponentBid) >=
	 * negotiationSession
	 * .getUtilitySpace().getUtility(opponentBidHistory.getBestBidInHistory()) -
	 * 0.01) { System.out.println("he offered me " + currentMaximumUtility + "
	 * we predict we can get at most " + predictMaximumUtility + "we concede now
	 * to avoid lower payoff due to conflict"); return true; } else {
	 * this.concedeToOpponent = true; return false; } } catch (Exception e) {
	 * System.out.println("exception in Method AcceptOpponentOffer"); return
	 * true; } //retrieve the opponent's biding history and utilize it } else if
	 * (currentMaximumUtility > this.utilitythreshold *
	 * Math.pow(this.discountingFactor,
	 * negotiationSession.getTimeline().getTime())) { try { //if the current
	 * offer is approximately as good as the best one in the history, then
	 * accept it. if
	 * (negotiationSession.getUtilitySpace().getUtility(opponentBid) >=
	 * negotiationSession
	 * .getUtilitySpace().getUtility(opponentBidHistory.getBestBidInHistory()) -
	 * 0.01) { return true; } else { System.out.println("test" +
	 * negotiationSession.getUtilitySpace().getUtility(opponentBid) +
	 * this.utilitythreshold); this.concedeToOpponent = true; return false; } }
	 * catch (Exception e) { System.out.println("exception in Method
	 * AcceptOpponentOffer"); return true; } } else { return false; } } }
	 */

	/**
	 * REMOVED AC private boolean TerminateCurrentNegotiation(Bid ownBid) {
	 * double currentUtility = 0; double nextRoundUtility = 0; double
	 * maximumUtility = 0; this.concedeToOpponent = false; try { currentUtility
	 * = this.reservationValue; nextRoundUtility =
	 * this.negotiationSession.getUtilitySpace().getUtility(ownBid);
	 * maximumUtility = this.MaximumUtility; } catch (Exception e) {
	 * System.out.println(e.getMessage() + "Exception in method
	 * TerminateCurrentNegotiation part 1"); }
	 * 
	 * if (currentUtility >= this.utilitythreshold || currentUtility >=
	 * nextRoundUtility) { return true; } else { //if the current reseravation
	 * utility with discount is larger than the predicted maximum utility with
	 * discount //then terminate the negotiation. double predictMaximumUtility =
	 * maximumUtility * this.discountingFactor; double currentMaximumUtility =
	 * this
	 * .negotiationSession.getUtilitySpace().getReservationValueWithDiscount(
	 * negotiationSession.getTimeline()); // System.out.println("the current
	 * reserved value is "+ this.reservationValue+" after discounting is
	 * "+currentMaximumUtility); if (currentMaximumUtility >
	 * predictMaximumUtility && negotiationSession.getTimeline().getTime() >
	 * this.concedeToDiscountingFactor) { return true; } else { return false; }
	 * } }
	 */

	/*
	 * estimate the number of rounds left before reaching the deadline @param
	 * opponent @return
	 */

	private int estimateRoundLeft(boolean opponent) {
		double round;
		if (opponent == true) {
			if (this.timeLeftBefore - this.timeLeftAfter > this.maximumTimeOfOpponent) {
				this.maximumTimeOfOpponent = this.timeLeftBefore - this.timeLeftAfter;
			}
		} else {
			if (this.timeLeftAfter - this.timeLeftBefore > this.maximumTimeOfOwn) {
				this.maximumTimeOfOwn = this.timeLeftAfter - this.timeLeftBefore;
			}
		}
		if (this.maximumTimeOfOpponent + this.maximumTimeOfOwn == 0) {
			System.out.println("divided by zero exception");
		}
		round = (this.totalTime - negotiationSession.getTimeline().getCurrentTime())
				/ (this.maximumTimeOfOpponent + this.maximumTimeOfOwn);
		// System.out.println("current time is " + timeline.getElapsedSeconds()
		// + "---" + round + "----" + this.maximumTimeOfOpponent);
		return ((int) (round));
	}

	/*
	 * pre-processing to save the computational time each round
	 */
	private void calculateBidsBetweenUtility() {
		BidIterator myBidIterator = new BidIterator(this.negotiationSession.getUtilitySpace().getDomain());

		try {
			// double maximumUtility =
			// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			double maximumUtility = this.MaximumUtility;
			double minUtility = this.minimumUtilityThreshold;
			int maximumRounds = (int) ((maximumUtility - minUtility) / 0.01);
			// initalization for each arraylist storing the bids between each
			// range
			for (int i = 0; i < maximumRounds; i++) {
				ArrayList<Bid> BidList = new ArrayList<Bid>();
				// BidList.add(this.bid_maximum_utility);
				this.bidsBetweenUtility.add(BidList);
			}
			this.bidsBetweenUtility.get(maximumRounds - 1).add(this.bid_maximum_utility);
			// note that here we may need to use some trick to reduce the
			// computation cost (to be checked later);
			// add those bids in each range into the corresponding arraylist
			int limits = 0;
			if (this.maximumOfBid < 20000) {
				while (myBidIterator.hasNext()) {
					Bid b = myBidIterator.next();
					for (int i = 0; i < maximumRounds; i++) {
						if (negotiationSession.getUtilitySpace().getUtility(b) <= (i + 1) * 0.01 + minUtility
								&& negotiationSession.getUtilitySpace().getUtility(b) >= i * 0.01 + minUtility) {
							this.bidsBetweenUtility.get(i).add(b);
							break;
						}
					}
					// limits++;
				}
			} else {
				while (limits <= 20000) {
					Bid b = this.RandomSearchBid();
					for (int i = 0; i < maximumRounds; i++) {
						if (negotiationSession.getUtilitySpace().getUtility(b) <= (i + 1) * 0.01 + minUtility
								&& negotiationSession.getUtilitySpace().getUtility(b) >= i * 0.01 + minUtility) {
							this.bidsBetweenUtility.get(i).add(b);
							break;
						}
					}
					limits++;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in calculateBidsBetweenUtility()");
			e.printStackTrace();
		}
	}

	private Bid RandomSearchBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();
		Bid bid = null;

		for (Issue lIssue : issues) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				int optionIndex = random.nextInt(lIssueDiscrete.getNumberOfValues());
				values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				int optionInd = random.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
				values.put(lIssueReal.getNumber(),
						new ValueReal(lIssueReal.getLowerBound()
								+ (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) * (double) (optionInd)
										/ (double) (lIssueReal.getNumberOfDiscretizationSteps())));
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				int optionIndex2 = lIssueInteger.getLowerBound()
						+ random.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
				values.put(lIssueInteger.getNumber(), new ValueInteger(optionIndex2));
				break;
			default:
				throw new Exception("issue type " + lIssue.getType() + " not supported");
			}
		}
		bid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
		return bid;
	}

	/*
	 * Get all the bids within a given utility range.
	 */
	private List<Bid> getBidsBetweenUtility(double lowerBound, double upperBound) {
		List<Bid> bidsInRange = new ArrayList<Bid>();
		try {
			int range = (int) ((upperBound - this.minimumUtilityThreshold) / 0.01);
			int initial = (int) ((lowerBound - this.minimumUtilityThreshold) / 0.01);
			// System.out.println(range+"---"+initial);
			for (int i = initial; i < range; i++) {
				bidsInRange.addAll(this.bidsBetweenUtility.get(i));
			}
			if (bidsInRange.isEmpty()) {
				bidsInRange.add(this.bid_maximum_utility);
			}
		} catch (Exception e) {
			System.out.println("Exception in getBidsBetweenUtility");
			e.printStackTrace();
		}
		return bidsInRange;
	}

	/*
	 * determine the lowest bound of our utility threshold based on the
	 * discounting factor we think that the minimum utility threshold should not
	 * be related with the discounting degree.
	 */
	private void chooseUtilityThreshold() {
		double discountingFactor = this.discountingFactor;
		if (discountingFactor >= 0.9) {
			this.minimumUtilityThreshold = 0;// this.MaximumUtility - 0.09;
		} else {
			// this.minimumUtilityThreshold = 0.85;
			this.minimumUtilityThreshold = 0;// this.MaximumUtility - 0.09;
		}
	}

	/*
	 * determine concede-to-time degree based on the discounting factor.
	 */

	private void chooseConcedeToDiscountingDegree() {
		double alpha = 0;
		double beta = 1.5;// 1.3;//this value controls the rate at which the
							// agent concedes to the discouting factor.
		// the larger beta is, the more the agent makes concesions.
		// if (utilitySpace.getDomain().getNumberOfPossibleBids() > 100) {
		/*
		 * if (this.maximumOfBid > 100) { beta = 2;//1.3; } else { beta = 1.5; }
		 */
		// the vaule of beta depends on the discounting factor (trade-off
		// between concede-to-time degree and discouting factor)
		if (this.discountingFactor > 0.75) {
			beta = 1.8;
		} else if (this.discountingFactor > 0.5) {
			beta = 1.5;
		} else {
			beta = 1.2;
		}
		alpha = Math.pow(this.discountingFactor, beta);
		this.concedeToDiscountingFactor = this.minConcedeToDiscountingFactor
				+ (1 - this.minConcedeToDiscountingFactor) * alpha;
		this.concedeToDiscountingFactor_original = this.concedeToDiscountingFactor;
		System.out.println("concedeToDiscountingFactor is " + this.concedeToDiscountingFactor + "current time is "
				+ negotiationSession.getTimeline().getTime());
	}

	/*
	 * receiveMessage the concede-to-time degree based on the predicted
	 * toughness degree of the opponent
	 */

	private void updateConcedeDegree() {
		double gama = 10;
		double weight = 0.1;
		double opponnetToughnessDegree = this.opponentBidHistory.getConcessionDegree();
		// this.concedeToDiscountingFactor =
		// this.concedeToDiscountingFactor_original * (1 +
		// opponnetToughnessDegree);
		this.concedeToDiscountingFactor = this.concedeToDiscountingFactor_original
				+ weight * (1 - this.concedeToDiscountingFactor_original) * Math.pow(opponnetToughnessDegree, gama);
		if (this.concedeToDiscountingFactor >= 1) {
			this.concedeToDiscountingFactor = 1;
		}
		// System.out.println("concedeToDiscountingFactor is " +
		// this.concedeToDiscountingFactor + "current time is " +
		// timeline.getTime() + "original concedetodiscoutingfactor is " +
		// this.concedeToDiscountingFactor_original);
	}

	@Override
	public String getName() {
		return "2012 - CUHKAgent";
	}

}
