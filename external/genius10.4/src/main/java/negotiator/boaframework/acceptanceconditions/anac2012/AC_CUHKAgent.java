package negotiator.boaframework.acceptanceconditions.anac2012;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import agents.anac.y2012.CUHKAgent.OpponentBidHistory;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
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
import negotiator.boaframework.sharedagentstate.anac2012.CUHKAgentSAS;

/**
 * This is the decoupled Acceptance Condition from CUHKAgent (ANAC2012). The
 * code was taken from the ANAC2012 CUHKAgent and adapted to work within the BOA
 * framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 *
 * @author Alex Dirkzwager
 * @version 31/10/12
 */
public class AC_CUHKAgent extends AcceptanceStrategy {

	private boolean activeHelper = false;
	private double reservationValue;
	private AdditiveUtilitySpace utilitySpace;
	private double discountingFactor;
	private double previousTime;
	private Actions nextAction;
	private BidDetails opponentBid = null;
	private double maximumOfBid;
	private OpponentBidHistory opponentBidHistory;
	private double minimumUtilityThreshold;
	private double concedeToDiscountingFactor_original;
	private double minConcedeToDiscountingFactor;
	private ArrayList<ArrayList<Bid>> bidsBetweenUtility;
	private Bid bid_maximum_utility;// the bid with the maximum utility over the
									// utility space.

	private Random random;
	private final boolean TEST_EQUIVALENCE = true;

	public AC_CUHKAgent() {
	}

	public AC_CUHKAgent(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		initializeAgent(negoSession, strat);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		initializeAgent(negoSession, strat);
	}

	public void initializeAgent(NegotiationSession negotiationSession, OfferingStrategy os) throws Exception {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = os;

		previousTime = 0;

		// checking if offeringStrategy SAS is a BRAMAgentSAS
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("CUHKAgent"))) {
			helper = new CUHKAgentSAS(negotiationSession);
			activeHelper = true;
		} else {
			helper = (CUHKAgentSAS) offeringStrategy.getHelper();
		}
		utilitySpace = (AdditiveUtilitySpace) negotiationSession.getUtilitySpace();
		this.reservationValue = 0;

		if (utilitySpace.getReservationValue() > 0) {
			this.reservationValue = utilitySpace.getReservationValue();
		}

		this.discountingFactor = 1;
		if (utilitySpace.getDiscountFactor() <= 1D && utilitySpace.getDiscountFactor() > 0D) {
			this.discountingFactor = utilitySpace.getDiscountFactor();
		}

		if (activeHelper) {
			try {
				maximumOfBid = this.utilitySpace.getDomain().getNumberOfPossibleBids();
				opponentBidHistory = new OpponentBidHistory();
				bidsBetweenUtility = new ArrayList<ArrayList<Bid>>();
				this.bid_maximum_utility = utilitySpace.getMaxUtilityBid();
				this.minConcedeToDiscountingFactor = 0.08;// 0.1;
				this.chooseUtilityThreshold();
				this.calculateBidsBetweenUtility();
				this.chooseConcedeToDiscountingDegree();

				this.opponentBidHistory.initializeDataStructures(utilitySpace.getDomain());
				((CUHKAgentSAS) helper).setTimeLeftAfter(negotiationSession.getTimeline().getCurrentTime());

				if (TEST_EQUIVALENCE) {
					random = new Random(100);
				} else {
					random = new Random();
				}

			} catch (Exception e) {
				System.out.println("initialization error" + e.getMessage());
			}
		}
	}

	@Override
	public Actions determineAcceptability() {
		if (activeHelper)
			nextAction = activeDetermineAcceptability();
		else
			nextAction = regularDetermineAcceptability();
		return nextAction;
	}

	private Actions activeDetermineAcceptability() {
		nextAction = Actions.Reject;
		try {
			double currentTime = negotiationSession.getTime();
			((CUHKAgentSAS) helper).addTimeInterval(currentTime - previousTime);
			previousTime = currentTime;

			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			((CUHKAgentSAS) helper).setTimeLeftBefore(negotiationSession.getTimeline().getCurrentTime());
			if (negotiationSession.getOpponentBidHistory().size() >= 1) {// the
																			// opponent
																			// propose
																			// first
																			// and
																			// we
																			// response
																			// secondly
				opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
				// receiveMessage opponent model first
				this.opponentBidHistory.updateOpponentModel(opponentBid.getBid(), utilitySpace.getDomain(),
						this.utilitySpace);
				this.updateConcedeDegree();
				// receiveMessage the estimation
				if (negotiationSession.getOwnBidHistory().isEmpty()) {
					// bid = utilitySpace.getMaxUtilityBid();
				} else {// other conditions
					if (((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, true) > 10) {// still
																							// have
																							// some
																							// rounds
																							// left
																							// to
																							// further
																							// negotiate
																							// (the
																							// major
																							// negotiation
																							// period)

						// System.out.println("Decoupled bid1: " + bid);
						// we expect that the negotiation is over once we select
						// a bid from the opponent's history.
						System.out.println("test2: " + ((CUHKAgentSAS) helper).isConcedeToOpponent());

						if (((CUHKAgentSAS) helper).isConcedeToOpponent() == true) {

							// System.out.println("we offer the best bid in the
							// history and the opponent should accept it");
							((CUHKAgentSAS) helper).setToughAgent(true);
							((CUHKAgentSAS) helper).setConcedeToOpponent(false);
						} else {
							((CUHKAgentSAS) helper).setToughAgent(false);
							// System.out.println("i propose " + debug +
							// " bid at time " + timeline.getTime());
						}

					} else {// this is the last chance and we concede by
							// providing the opponent the best offer he ever
							// proposed to us
						// in this case, it corresponds to an opponent whose
						// decision time is short
						if (negotiationSession.getTimeline().getTime() > 0.9985
								&& ((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, true) < 5) {
							// bid =
							// opponentBidHistory.chooseBestFromHistory(this.utilitySpace);
							// this is specially designed to avoid that we got
							// very low utility by searching between an
							// acceptable range (when the domain is small)

							if (offeringStrategy.getNextBid().getMyUndiscountedUtil() < 0.85) {

								// if the candidate bids do not exsit and also
								// the deadline is approaching in next round, we
								// concede.
								// if (candidateBids.size() == 1 &&
								// timeline.getTime()>0.9998) {
								// we have no chance to make a new proposal
								// before the deadline
								if (((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, true) < 2) {
									// bid =
									// opponentBidHistory.getBestBidInHistory();
									// System.out.printlned bid3: " + bid);

									// System.out.println("test I " +
									// utilitySpace.getUtility(bid));
								}

							}

							if (((CUHKAgentSAS) helper).isToughAgent() == true) {
								nextAction = Actions.Accept;
								System.out.println(
										"the opponent is tough and the deadline is approching thus we accept the offer");
							}

						}
					}
				}
			}
			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			((CUHKAgentSAS) helper).setTimeLeftAfter(negotiationSession.getTimeline().getCurrentTime());
			((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, false);// receiveMessage
																		// the
																		// estimation
			// System.out.println(this.utilitythreshold + "-***-----" +
			// this.timeline.getElapsedSeconds());
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			System.out.println(((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, false));
			// action = new Accept(getAgentID()); // accept if anything goes
			// wrong.
			// bidToOffer = new EndNegotiation(getAgentID()); //terminate if
			// anything goes wrong.
		}
		return nextAction;
	}

	private Actions regularDetermineAcceptability() {
		if (activeHelper) {
			double currentTime = negotiationSession.getTime();
			((CUHKAgentSAS) helper).addTimeInterval(currentTime - previousTime);
			previousTime = currentTime;

			if (((CUHKAgentSAS) helper).estimateTheRoundsLeft(true, true) > 10) {// still
																					// have
																					// some
																					// rounds
																					// left
																					// to
																					// further
																					// negotiate
																					// (the
																					// major
																					// negotiation
																					// period)

				if (((CUHKAgentSAS) helper).isConcedeToOpponent() == true) {
					((CUHKAgentSAS) helper).setToughAgent(true);
					((CUHKAgentSAS) helper).setConcedeToOpponent(false);
				} else {
					((CUHKAgentSAS) helper).setToughAgent(false);
				}
			}
		}

		Bid opponentBid = negotiationSession.getOpponentBidHistory().getLastBid();
		// Double check this corresponds with the original every time
		Bid bid = offeringStrategy.getNextBid().getBid();
		int roundsLeft;
		if (activeHelper) {
			roundsLeft = ((CUHKAgentSAS) helper).estimateTheRoundsLeft(true, true);
		} else {
			roundsLeft = ((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, true);
		}
		if (roundsLeft > 10) {// still have some rounds left to further
								// negotiate (the major negotiation period)
			Boolean IsAccept = AcceptOpponentOffer(opponentBid, bid);
			Boolean IsTerminate = TerminateCurrentNegotiation(bid);

			if (IsAccept && !IsTerminate) {
				System.out.println("accept the offer");
				return Actions.Accept;
			} else if (IsTerminate && !IsAccept) {
				System.out.println("we determine to terminate the negotiation");
				return Actions.Break;
			} else if (IsAccept && IsTerminate) {
				try {
					if (this.utilitySpace.getUtility(opponentBid) > this.reservationValue) {
						System.out.println("we accept the offer RANDOMLY");
						return Actions.Accept;
					} else {
						System.out.println("we determine to terminate the negotiation RANDOMLY");
						return Actions.Break;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

		} else {

			if (activeHelper) {
				roundsLeft = ((CUHKAgentSAS) helper).estimateTheRoundsLeft(true, true);
			} else {
				roundsLeft = ((CUHKAgentSAS) helper).estimateTheRoundsLeft(false, true);
			}

			if (negotiationSession.getTime() > 0.9985 && roundsLeft < 5) {
				Boolean IsAccept = AcceptOpponentOffer(opponentBid, bid);
				Boolean IsTerminate = TerminateCurrentNegotiation(bid);
				if (IsAccept && !IsTerminate) {
					System.out.println("accept the offer");
					return Actions.Accept;
				} else if (IsTerminate && !IsAccept) {
					System.out.println("we determine to terminate the negotiation");
					return Actions.Break;
				} else if (IsTerminate && IsAccept) {
					try {
						if (utilitySpace.getUtility(opponentBid) > this.reservationValue) {
							System.out.println("we accept the offer RANDOMLY");
							return Actions.Accept;
						} else {
							System.out.println("we determine to terminate the negotiation RANDOMLY");
							return Actions.Break;
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				} else {
					if (((CUHKAgentSAS) helper).isToughAgent() == true) {
						System.out.println(
								"the opponent is tough and the deadline is approching thus we accept the offer");
						return Actions.Accept;
					}
				}
			} else {
				Boolean IsAccept = AcceptOpponentOffer(opponentBid, bid);
				Boolean IsTerminate = TerminateCurrentNegotiation(bid);
				if (IsAccept && !IsTerminate) {
					System.out.println("accept the offer");
					return Actions.Accept;
				} else if (IsTerminate && !IsAccept) {
					System.out.println("we determine to terminate the negotiation");
					return Actions.Break;
				} else if (IsAccept && IsTerminate) {
					try {
						if (utilitySpace.getUtility(opponentBid) > this.reservationValue) {
							System.out.println("we accept the offer RANDOMLY");
							return Actions.Accept;
						} else {
							System.out.println("we determine to terminate the negotiation RANDOMLY");
							return Actions.Break;
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}

		}

		return Actions.Reject;
	}

	/*
	 * decide whether to accept the current offer or not
	 */
	private boolean AcceptOpponentOffer(Bid opponentBid, Bid ownBid) {
		double currentUtility = 0;
		double nextRoundUtility = 0;
		double maximumUtility = 0;
		((CUHKAgentSAS) helper).setConcedeToOpponent(false);
		try {
			currentUtility = this.utilitySpace.getUtility(opponentBid);
			if (activeHelper) {
				maximumUtility = utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			} else {
				maximumUtility = ((CUHKAgentSAS) helper).getMaximumUtility();// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			}
		} catch (Exception e) {
			System.out.println(e.getMessage() + "Exception in method AcceptOpponentOffer part 1");
		}
		try {
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
		} catch (Exception e) {
			System.out.println(e.getMessage() + "Exception in method AcceptOpponentOffer part 2");
		}
		// System.out.println(this.utilitythreshold +"at time "+
		// timeline.getTime());
		if (currentUtility >= ((CUHKAgentSAS) helper).getUtilitythreshold() || currentUtility >= nextRoundUtility) {
			return true;
		} else {
			// if the current utility with discount is larger than the predicted
			// maximum utility with discount
			// then accept it.
			double predictMaximumUtility = maximumUtility * this.discountingFactor;
			// double currentMaximumUtility =
			// this.utilitySpace.getUtilityWithDiscount(opponentBidHistory.chooseBestFromHistory(utilitySpace),
			// timeline);
			double currentMaximumUtility = this.utilitySpace.getUtilityWithDiscount(
					negotiationSession.getOpponentBidHistory().getBestBidDetails().getBid(),
					negotiationSession.getTimeline());
			if (currentMaximumUtility > predictMaximumUtility
					&& negotiationSession.getTime() > ((CUHKAgentSAS) helper).getConcedeToDiscountingFactor()) {
				try {
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace.getUtility(opponentBid) >= utilitySpace.getUtility(
							negotiationSession.getOpponentBidHistory().getBestBidDetails().getBid()) - 0.01) {
						System.out.println("he offered me " + currentMaximumUtility + " we predict we can get at most "
								+ predictMaximumUtility + "we concede now to avoid lower payoff due to conflict");
						return true;
					} else {
						((CUHKAgentSAS) helper).setConcedeToOpponent(true);
						return false;
					}
				} catch (Exception e) {
					System.out.println("exception in Method AcceptOpponentOffer");
					return true;
				}
				// retrieve the opponent's biding history and utilize it
			} else if (currentMaximumUtility > ((CUHKAgentSAS) helper).getUtilitythreshold()
					* Math.pow(this.discountingFactor, negotiationSession.getTime())) {
				try {
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace.getUtility(opponentBid) >= utilitySpace.getUtility(
							negotiationSession.getOpponentBidHistory().getBestBidDetails().getBid()) - 0.01) {
						return true;
					} else {
						System.out.println("test" + utilitySpace.getUtility(opponentBid)
								+ ((CUHKAgentSAS) helper).getUtilitythreshold());
						((CUHKAgentSAS) helper).setConcedeToOpponent(true);
						return false;
					}
				} catch (Exception e) {
					System.out.println("exception in Method AcceptOpponentOffer");
					return true;
				}
			} else {
				return false;
			}
		}
	}

	/*
	 * decide whether or not to terminate now
	 */
	private boolean TerminateCurrentNegotiation(Bid ownBid) {
		double currentUtility = 0;
		double nextRoundUtility = 0;
		double maximumUtility = 0;
		((CUHKAgentSAS) helper).setConcedeToOpponent(false);
		try {
			currentUtility = this.reservationValue;
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
			maximumUtility = ((CUHKAgentSAS) helper).getMaximumUtility();
		} catch (Exception e) {
			System.out.println(e.getMessage() + "Exception in method TerminateCurrentNegotiation part 1");
		}

		if (currentUtility >= ((CUHKAgentSAS) helper).getUtilitythreshold() || currentUtility >= nextRoundUtility) {
			return true;
		} else {
			// if the current reseravation utility with discount is larger than
			// the predicted maximum utility with discount
			// then terminate the negotiation.
			double predictMaximumUtility = maximumUtility * this.discountingFactor;
			double currentMaximumUtility = this.utilitySpace
					.getReservationValueWithDiscount(negotiationSession.getTimeline());
			// System.out.println("the current reserved value is "+
			// this.reservationValue+" after discounting is
			// "+currentMaximumUtility);
			if (currentMaximumUtility > predictMaximumUtility
					&& negotiationSession.getTime() > ((CUHKAgentSAS) helper).getConcedeToDiscountingFactor()) {
				return true;
			} else {
				return false;
			}
		}
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
		((CUHKAgentSAS) helper).setConcedeToDiscountingFactor(
				this.minConcedeToDiscountingFactor + (1 - this.minConcedeToDiscountingFactor) * alpha);
		this.concedeToDiscountingFactor_original = ((CUHKAgentSAS) helper).getConcedeToDiscountingFactor();
		System.out.println("concedeToDiscountingFactor is " + ((CUHKAgentSAS) helper).getConcedeToDiscountingFactor()
				+ "current time is " + negotiationSession.getTimeline().getTime());
	}

	/*
	 * pre-processing to save the computational time each round
	 */
	private void calculateBidsBetweenUtility() {
		BidIterator myBidIterator = new BidIterator(this.utilitySpace.getDomain());

		try {
			// double maximumUtility =
			// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			double maximumUtility = (((CUHKAgentSAS) helper).getMaximumUtility());
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
						if (utilitySpace.getUtility(b) <= (i + 1) * 0.01 + minUtility
								&& utilitySpace.getUtility(b) >= i * 0.01 + minUtility) {
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
						if (utilitySpace.getUtility(b) <= (i + 1) * 0.01 + minUtility
								&& utilitySpace.getUtility(b) >= i * 0.01 + minUtility) {
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
		List<Issue> issues = utilitySpace.getDomain().getIssues();
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
		bid = new Bid(utilitySpace.getDomain(), values);
		return bid;
	}

	private void updateConcedeDegree() {
		double gama = 10;
		double weight = 0.1;
		double opponnetToughnessDegree = this.opponentBidHistory.getConcessionDegree();
		// this.concedeToDiscountingFactor =
		// this.concedeToDiscountingFactor_original * (1 +
		// opponnetToughnessDegree);
		((CUHKAgentSAS) helper).setConcedeToDiscountingFactor(this.concedeToDiscountingFactor_original
				+ weight * (1 - this.concedeToDiscountingFactor_original) * Math.pow(opponnetToughnessDegree, gama));
		if (((CUHKAgentSAS) helper).getConcedeToDiscountingFactor() >= 1) {
			((CUHKAgentSAS) helper).setConcedeToDiscountingFactor(1);
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
