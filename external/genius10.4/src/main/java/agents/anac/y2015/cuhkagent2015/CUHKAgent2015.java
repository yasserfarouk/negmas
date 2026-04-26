package agents.anac.y2015.cuhkagent2015;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
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
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 *
 * @author Tom
 */
public class CUHKAgent2015 extends AbstractNegotiationParty {
	boolean debug = false;

	private final double totalTime = 180;
	// private Action ActionOfOpponent = null;
	private double maximumOfBid;
	private OwnBidHistory ownBidHistory;
	private OpponentBidHistory opponentBidHistory1;
	private OpponentBidHistory opponentBidHistory2;
	private double minimumUtilityThreshold;
	private double AvgUtilitythreshold;
	private double AvgConcedeTime;
	private double MaximumUtility;
	private int limit;// the maximimum number of trials for each searching range
	private double numberOfRounds;
	private double timeLeftBefore;
	private double timeLeftAfter;
	private double maximumTimeOfOpponent;
	private double maximumTimeOfOwn;
	private double discountingFactor;
	private double concedeToDiscountingFactor;
	private double concedeToDiscountingFactor_original;
	private double minConcedeToDiscountingFactor;
	private ArrayList<ArrayList<Bid>> bidsBetweenUtility;
	private double previousToughnessDegree;
	private boolean concedeToOpponent;
	private boolean toughAgent; // if we propose a bid that was proposed by the
								// opponnet, then it should be accepted.
	private double alpha1;// the larger alpha is, the more tough the agent is.
	private Bid bid_maximum_utility;// the bid with the maximum utility over the
									// utility space.
	private double reservationValue;

	/* UpdateConcede function's Constant */
	private double k = 1;
	private double N = 0;
	private int FirstTimeInterval;
	private int SecondTimeInterval;
	private int ThirdTimeInterval;

	/* Agent 1 Analysis */
	private double utility_maximum_from_opponent_Session1_Agent1 = 0;
	private double utility_maximum_from_opponent_Session2_Agent1 = 0;
	private double utility_maximum_from_opponent_Session3_Agent1 = 0;
	private double count_Agent1 = -1;
	private double concedePartOfdiscountingFactor_Agent1 = 0;
	private double concedePartOfOppBehaviour_Agent1 = 0;
	private double minThreshold_Agent1;
	private double oppFirstBidUtility_Agent1;
	private Bid oppFirstBid_Agent1;
	private double relstdevUtility_Agent1;
	private boolean startConcede_Agent1 = false;
	private double relcountUpperBoundMid1_Agent1;
	private double relcountLowerBoundMid1_Agent1;
	private double relcountUpperBoundMid2_Agent1;
	private double relcountLowerBoundMid2_Agent1;
	private double relsumUtility_Agent1 = 0;
	private double relcountUtility_Agent1 = 0;
	private double relavgUtility_Agent1 = 0;
	private double relSqRootOfAvgUtility_Agent1 = 0;
	private double relcountUtilityInSmallSD_Agent1 = 0;
	private double utility_FirstMaximum_Agent1 = 0;
	private double utility_SecondMaximum_Agent1 = 0;
	private double midPointOfSlopeSessionMax1_Agent1 = 0;
	private double midPointOfSlopeSessionMax2_Agent1 = 0;
	private double slopeOfSlopeOfSessionMax1_Agent1 = 0;
	private double slopeOfSlopeOfSessionMax2_Agent1 = 0;
	private boolean IsOppFirstBid_Agent1;
	private Bid oppPreviousBid_Agent1;
	private Bid minAcceptableBid;
	private double opponentFirstBidUtility_Agent;
	private double concedeTime_Agent1;
	private double MinimumUtility_Agent1;
	private double utilitythreshold_Agent1;

	/* Agent 2 Analysis */
	private double utility_maximum_from_opponent_Session1_Agent2 = 0;
	private double utility_maximum_from_opponent_Session2_Agent2 = 0;
	private double utility_maximum_from_opponent_Session3_Agent2 = 0;
	private double count_Agent2 = -1;
	private double concedePartOfdiscountingFactor_Agent2 = 0;
	private double concedePartOfOppBehaviour_Agent2 = 0;
	private double minThreshold_Agent2;
	private Bid oppFirstBid_Agent2;
	private double oppFirstBidUtility_Agent2;
	private double relstdevUtility_Agent2;
	private boolean startConcede_Agent2 = false;
	private double relcountUpperBoundMid1_Agent2;
	private double relcountLowerBoundMid1_Agent2;
	private double relcountUpperBoundMid2_Agent2;
	private double relcountLowerBoundMid2_Agent2;
	private double relsumUtility_Agent2 = 0;
	private double relcountUtility_Agent2 = 0;
	private double relavgUtility_Agent2 = 0;
	private double relSqRootOfAvgUtility_Agent2 = 0;
	private double relcountUtilityInSmallSD_Agent2 = 0;
	private double utility_FirstMaximum_Agent2 = 0;
	private double utility_SecondMaximum_Agent2 = 0;
	private double midPointOfSlopeSessionMax1_Agent2 = 0;
	private double midPointOfSlopeSessionMax2_Agent2 = 0;
	private double slopeOfSlopeOfSessionMax1_Agent2 = 0;
	private double slopeOfSlopeOfSessionMax2_Agent2 = 0;
	private boolean IsOppFirstBid_Agent2;
	private Bid oppPreviousBid_Agent2;
	private double concedeTime_Agent2;
	private double MinimumUtility_Agent2;
	private double utilitythreshold_Agent2;

	private double MinimumUtility;
	private AgentID partyId;
	private final double domainthreshold = 2000;
	private final double totaltime = 180;
	private Action ActionOfOpponent = null;
	private double NumberOfBid;
	private final int NumberofDivisions = 20;
	private double SpacebetweenDivisions;
	private int CurrentrangeNumber;
	private int maximumOfRound;
	private final int limitValue = 5;// 10;
	private double delta; // searching range
	private double concedeTime_original;
	private double minConcedeTime;
	private List<Issue> Issues;
	private Bid bid_minimum_utility;// the bid with the minimum utility over the
									// utility space.
	private Bid mylastBid = this.bid_maximum_utility;
	private ArrayList<Bid> OppBids1;
	private ArrayList<Bid> OppBids2;
	private int storingcapacity = 500;

	private double nextbidutil;
	private double mindistance;
	private double averagedistance;
	private int test = 0;
	private int test2 = 0;
	private int test3 = 0;
	private int except = 0;
	private int size;
	private double minUtilityUhreshold = 1.1;
	private double oppbid;
	private int countOpp;
	private int EachRoundCount;
	private Bid myLastBid;

	PrintStream file, SysOut;
	String fileName;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		try {
			this.NumberOfBid = this.utilitySpace.getDomain()
					.getNumberOfPossibleBids();
			this.ownBidHistory = new OwnBidHistory();
			this.opponentBidHistory1 = new OpponentBidHistory();
			this.opponentBidHistory2 = new OpponentBidHistory();
			this.Issues = this.utilitySpace.getDomain().getIssues();
			this.bidsBetweenUtility = new ArrayList<ArrayList<Bid>>();
			this.OppBids1 = new ArrayList<Bid>();
			this.OppBids2 = new ArrayList<Bid>();
			for (int i = 0; i < this.storingcapacity; i++) {
				this.OppBids1.add(i, null);
				this.OppBids2.add(i, null);
			}
			this.bid_maximum_utility = this.utilitySpace.getMaxUtilityBid();
			this.bid_minimum_utility = this.utilitySpace.getMinUtilityBid();
			this.MaximumUtility = this.utilitySpace
					.getUtility(this.bid_maximum_utility);
			this.reservationValue = utilitySpace.getReservationValue();
			this.MinimumUtility_Agent1 = this.reservationValue;
			this.MinimumUtility_Agent2 = this.reservationValue;
			this.delta = 0.01; // searching range
			this.limit = 10;
			this.previousToughnessDegree = 0;
			this.numberOfRounds = 0;
			this.timeLeftAfter = 0;
			this.timeLeftBefore = 0;
			this.maximumTimeOfOpponent = 0;
			this.maximumTimeOfOwn = 0;
			this.minConcedeTime = 0.08;// 0.1;
			this.discountingFactor = 1;
			if (utilitySpace.getDiscountFactor() <= 1D
					&& utilitySpace.getDiscountFactor() > 0D) {
				this.discountingFactor = utilitySpace.getDiscountFactor();
			}
			this.calculateBidsBetweenUtility();
			if (this.discountingFactor <= 0.5D) {
				this.chooseConcedeToDiscountingDegree();
			}
			if (debug) {
				fileName = "plotData" + hashCode() + ".txt";
				file = new PrintStream(new FileOutputStream(fileName, false));
				file.println("\n");
			}
			this.opponentBidHistory1
					.initializeDataStructures(utilitySpace.getDomain());
			this.opponentBidHistory2
					.initializeDataStructures(utilitySpace.getDomain());
			this.timeLeftAfter = timeline.getCurrentTime();
			this.concedeToOpponent = false;
			this.toughAgent = false;
			this.alpha1 = 2;

			if (this.NumberOfBid < 100) {
				this.k = 1;
			} else if (this.NumberOfBid < 1000) {
				this.k = 2;
			} else if (this.NumberOfBid < 10000) {
				this.k = 3;
			} else {
				this.k = 4;
			}
			this.concedeTime_Agent1 = 0.5;
			this.concedeTime_Agent2 = 0.5;
			this.oppFirstBid_Agent1 = null;
			this.oppFirstBidUtility_Agent1 = 0;
			this.oppFirstBid_Agent2 = null;
			this.oppFirstBidUtility_Agent2 = 0;
			this.IsOppFirstBid_Agent1 = true;
			this.IsOppFirstBid_Agent2 = true;
			this.FirstTimeInterval = 20;
			this.SecondTimeInterval = 40;
			this.ThirdTimeInterval = 60;
			this.utilitythreshold_Agent1 = 0.5;
			this.utilitythreshold_Agent2 = 0.5;
			this.AvgUtilitythreshold = 0.5;
			this.AvgConcedeTime = 0.5;
			this.EachRoundCount = 0;
			this.myLastBid = this.bid_maximum_utility;
		} catch (Exception e) {
			System.out.println("initialization error" + e.getMessage());
		}
	}

	public String getVersion() {
		return "CUHKAgent2015";
	}

	public String getName() {
		return "CUHKAgent2015 Test Temp";
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		Action action = null;
		this.EachRoundCount = this.countOpp;
		try {
			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			this.timeLeftBefore = timeline.getCurrentTime();
			Bid bid = null;
			// we propose first and propose the bid with maximum utility
			if ((!possibleActions.contains(Accept.class))
					|| (this.oppFirstBid_Agent1 == null)
					|| (this.oppFirstBid_Agent2 == null)) {
				bid = this.bid_maximum_utility;
				action = new Offer(getPartyId(), bid);
			}
			// else if (ActionOfOpponent instanceof Offer)
			else {
				// the opponent propose first and we response secondly
				// update opponent model first
				this.opponentBidHistory1.updateOpponentModel(
						this.oppPreviousBid_Agent1, utilitySpace.getDomain(),
						(AdditiveUtilitySpace) this.utilitySpace);
				this.opponentBidHistory2.updateOpponentModel(
						this.oppPreviousBid_Agent2, utilitySpace.getDomain(),
						(AdditiveUtilitySpace) this.utilitySpace);
				if (this.discountingFactor == 1) {
					this.updateConcedeDegree_nonDiscountingFactor_Agent1();
					this.updateConcedeDegree_nonDiscountingFactor_Agent2();
				} else if (this.discountingFactor <= 0.5) {
					this.updateConcedeDegree_smallDiscountingFactor();
				} else {
					this.updateConcedeDegree_largeDiscountingFactor_Agent1();
					this.updateConcedeDegree_largeDiscountingFactor_Agent2();
				}
				// update the estimation
				if (ownBidHistory.numOfBidsProposed() == 0) {
					bid = this.bid_maximum_utility;
					action = new Offer(getPartyId(), bid);
				} else { // other conditions
					if (estimateRoundLeft(true) > 10) {
						// still have some rounds left to further negotiate (the
						// major negotiation period)
						bid = BidToOffer_original();
						Boolean IsAccept = AcceptOpponentOffer(
								this.oppPreviousBid_Agent2, bid);
						IsAccept = OtherAcceptCondition(IsAccept);

						Boolean IsTerminate = TerminateCurrentNegotiation(bid);
						if (IsAccept && !IsTerminate) {
							action = new Accept(getPartyId(),
									((ActionWithBid) getLastReceivedAction())
											.getBid());
							System.out.println("accept the offer");
						} else if (IsTerminate && !IsAccept) {
							action = new EndNegotiation(getPartyId());
							System.out.println(
									"we determine to terminate the negotiation");
						} else if (IsAccept && IsTerminate) {
							if (this.utilitySpace.getUtility(
									this.oppPreviousBid_Agent2) > this.reservationValue) {
								action = new Accept(getPartyId(),
										((ActionWithBid) getLastReceivedAction())
												.getBid());
								System.out.println(
										"we accept the offer RANDOMLY");
							} else {
								action = new EndNegotiation(getPartyId());
								System.out.println(
										"we determine to terminate the negotiation RANDOMLY");
							}
						} else {
							// we expect that the negotiation is over once we
							// select a bid from the opponent's history.
							if (this.concedeToOpponent == true) {
								// bid =
								// opponentBidHistory.chooseBestFromHistory(this.utilitySpace);
								Bid bid1 = opponentBidHistory1
										.getBestBidInHistory();
								Bid bid2 = opponentBidHistory2
										.getBestBidInHistory();
								if (this.utilitySpace
										.getUtility(bid1) > this.utilitySpace
												.getUtility(bid2)) {
									bid = bid2;
								} else if (this.utilitySpace
										.getUtility(bid1) == this.utilitySpace
												.getUtility(bid2)
										&& Math.random() < 0.5) {
									bid = bid2;
								} else {
									bid = bid1;
								}
								action = new Offer(getPartyId(), bid);
								// System.out.println("we offer the best bid in
								// the history and the opponent should accept
								// it");
								this.toughAgent = true;
								this.concedeToOpponent = false;
							} else {
								action = new Offer(getPartyId(), bid);
								this.toughAgent = false;
								// System.out.println("i propose " + debug +
								// " bid at time " + timeline.getTime());
							}
						}
					} else {// this is the last chance and we concede by
							// providing the opponent the best offer he ever
							// proposed to us
							// in this case, it corresponds to an opponent whose
							// decision time is short

						if (timeline.getTime() > 0.9985
								&& estimateRoundLeft(true) < 5) {
							bid = BidToOffer_original();

							Boolean IsAccept = AcceptOpponentOffer(
									this.oppPreviousBid_Agent2, bid);
							IsAccept = OtherAcceptCondition(IsAccept);

							Boolean IsTerminate = TerminateCurrentNegotiation(
									bid);
							if (IsAccept && !IsTerminate) {
								action = new Accept(getPartyId(),
										((ActionWithBid) getLastReceivedAction())
												.getBid());
								System.out.println("accept the offer");
							} else if (IsTerminate && !IsAccept) {
								action = new EndNegotiation(getPartyId());
								System.out.println(
										"we determine to terminate the negotiation");
							} else if (IsTerminate && IsAccept) {
								if (this.utilitySpace.getUtility(
										this.oppPreviousBid_Agent2) > this.reservationValue) {
									action = new Accept(getPartyId(),
											((ActionWithBid) getLastReceivedAction())
													.getBid());
									System.out.println(
											"we accept the offer RANDOMLY");
								} else {
									action = new EndNegotiation(getPartyId());
									System.out.println(
											"we determine to terminate the negotiation RANDOMLY");
								}
							} else {
								if (this.toughAgent == true) {
									action = new Accept(getPartyId(),
											((ActionWithBid) getLastReceivedAction())
													.getBid());
									System.out.println(
											"the opponent is tough and the deadline is approching thus we accept the offer");
								} else {
									action = new Offer(getPartyId(), bid);
									// this.toughAgent = true;
									System.out.println(
											"this is really the last chance"
													+ bid.toString()
													+ " with utility of "
													+ utilitySpace
															.getUtility(bid));
								}
							}
							// in this case, it corresponds to the situation
							// that we encounter an opponent who needs more
							// computation to make decision each round
						} else {// we still have some time to negotiate,
								// and be tough by sticking with the lowest one
								// in previous offer history.
								// we also have to make the decisin fast to
								// avoid reaching the deadline before the
								// decision is made
								// bid =
								// ownBidHistory.GetMinBidInHistory();//reduce
								// the computational cost
							bid = BidToOffer_original();
							// System.out.println("test----------------------------------------------------------"
							// + timeline.getTime());
							Boolean IsAccept = AcceptOpponentOffer(
									this.oppPreviousBid_Agent2, bid);
							IsAccept = OtherAcceptCondition(IsAccept);

							Boolean IsTerminate = TerminateCurrentNegotiation(
									bid);
							if (IsAccept && !IsTerminate) {
								action = new Accept(getPartyId(),
										((ActionWithBid) getLastReceivedAction())
												.getBid());
								System.out.println("accept the offer");
							} else if (IsTerminate && !IsAccept) {
								action = new EndNegotiation(getPartyId());
								System.out.println(
										"we determine to terminate the negotiation");
							} else if (IsAccept && IsTerminate) {
								if (this.utilitySpace.getUtility(
										this.oppPreviousBid_Agent2) > this.reservationValue) {
									action = new Accept(getPartyId(),
											((ActionWithBid) getLastReceivedAction())
													.getBid());
									System.out.println(
											"we accept the offer RANDOMLY");
								} else {
									action = new EndNegotiation(getPartyId());
									System.out.println(
											"we determine to terminate the negotiation RANDOMLY");
								}
							} else {
								action = new Offer(getPartyId(), bid);
								// System.out.println("we have to be tough now"
								// + bid.toString() + " with utility of " +
								// utilitySpace.getUtility(bid));
							}
						}
					}
				}
			}
			this.myLastBid = bid;
			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			this.ownBidHistory.addBid(bid, (AdditiveUtilitySpace) utilitySpace);
			this.timeLeftAfter = timeline.getCurrentTime();
			if (this.timeLeftAfter
					- this.timeLeftBefore > this.maximumTimeOfOwn) {
				this.maximumTimeOfOwn = this.timeLeftAfter
						- this.timeLeftBefore;
			} // update the estimation
				// System.out.println(this.utilitythreshold + "-***-----" +
				// this.timeline.getElapsedSeconds());
			if (debug) {
				try {
					SysOut = System.out;
					file = new PrintStream(
							new FileOutputStream(fileName, true));
					System.setOut(file);
				} catch (FileNotFoundException ex) {

				}
				System.out.println(Double.toString(this.timeline.getTime())
						+ "," + Double.toString(this.AvgUtilitythreshold) + ","
						+ Double.toString(this.minUtilityUhreshold));
				System.setOut(SysOut);
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			System.out.println(estimateRoundLeft(false));
			action = new EndNegotiation(getPartyId()); // terminate if anything
														// goes wrong.
		}
		return action;
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {
		super.receiveMessage(sender, opponentAction);
		// sender.getClass().getName();
		this.ActionOfOpponent = opponentAction;

		this.countOpp = this.countOpp + 1;
		if (this.countOpp == this.EachRoundCount + 1)// Next Agent w.r.t. us
		{
			if ((this.ActionOfOpponent instanceof Offer)) {
				if (this.oppFirstBid_Agent1 == null) {
					oppFirstBid_Agent1 = ((Offer) this.ActionOfOpponent)
							.getBid();
				}
				for (int i = 0; i < this.storingcapacity - 1; i++) {
					this.OppBids1.set(i, this.OppBids1.get(i + 1));
				}
				this.oppPreviousBid_Agent1 = ((Offer) this.ActionOfOpponent)
						.getBid();
				this.OppBids1.set(this.storingcapacity - 1,
						this.oppPreviousBid_Agent1);
			} else if ((this.ActionOfOpponent instanceof Accept)) {
				this.oppPreviousBid_Agent1 = this.myLastBid;
			}
		} else if (this.countOpp == this.EachRoundCount + 2)// Previous Agent
															// w.r.t. us
		{
			if ((this.ActionOfOpponent instanceof Offer)) {
				if (this.oppFirstBid_Agent2 == null) {
					oppFirstBid_Agent2 = ((Offer) this.ActionOfOpponent)
							.getBid();
				}
				for (int i = 0; i < this.storingcapacity - 1; i++) {
					this.OppBids2.set(i, this.OppBids2.get(i + 1));
				}
				this.oppPreviousBid_Agent2 = ((Offer) this.ActionOfOpponent)
						.getBid();
				this.OppBids2.set(this.storingcapacity - 1,
						this.oppPreviousBid_Agent2);
			} else if ((this.ActionOfOpponent instanceof Accept)) {
				this.oppPreviousBid_Agent2 = this.oppPreviousBid_Agent1;
			}
		}
	}

	private double BidToOffer() {
		try {

			double maximumOfBid = this.MaximumUtility;// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			AvgConcedeTime = (this.concedeTime_Agent2 + this.concedeTime_Agent1)
					/ 2;

			if (timeline.getTime() <= AvgConcedeTime) {
				if (this.discountingFactor <= 0.5) {
					this.minThreshold_Agent1 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.concedeTime_Agent1);
					this.minThreshold_Agent2 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.concedeTime_Agent2);
					this.utilitythreshold_Agent1 = maximumOfBid
							- (maximumOfBid - this.minThreshold_Agent1)
									* Math.pow(
											(timeline.getTime()
													/ this.concedeTime_Agent1),
											alpha1);
					this.utilitythreshold_Agent2 = maximumOfBid
							- (maximumOfBid - this.minThreshold_Agent2)
									* Math.pow(
											(timeline.getTime()
													/ this.concedeTime_Agent2),
											alpha1);
				} else {
					this.utilitythreshold_Agent1 = this.minThreshold_Agent1
							+ (maximumOfBid - this.minThreshold_Agent1) * (1
									- Math.sin((Math.PI / 2) * (timeline
											.getTime() / this.concedeTime_Agent1)));
					this.utilitythreshold_Agent2 = this.minThreshold_Agent2
							+ (maximumOfBid - this.minThreshold_Agent2) * (1
									- Math.sin((Math.PI / 2) * (timeline
											.getTime() / this.concedeTime_Agent2)));
				}
			} else {
				if (this.discountingFactor <= 0.5) {
					this.utilitythreshold_Agent1 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									timeline.getTime());
					this.utilitythreshold_Agent2 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									timeline.getTime());
				} else {
					this.utilitythreshold_Agent1 = this.minThreshold_Agent1
							+ (maximumOfBid - this.minThreshold_Agent1)
									/ (1 - concedeTime_Agent1)
									* Math.pow(
											(timeline.getTime()
													- this.concedeTime_Agent1),
											this.discountingFactor);
					this.utilitythreshold_Agent2 = this.minThreshold_Agent2
							+ (maximumOfBid - this.minThreshold_Agent2)
									/ (1 - concedeTime_Agent2)
									* Math.pow(
											(timeline.getTime()
													- this.concedeTime_Agent2),
											this.discountingFactor);
				}
			}
			this.AvgUtilitythreshold = (this.utilitythreshold_Agent2
					+ this.utilitythreshold_Agent1) / 2;

			if (this.AvgUtilitythreshold > MaximumUtility) {
				this.AvgUtilitythreshold = MaximumUtility;
			}

			boolean accept = false;
			int count = 0;
			while (!accept) {
				// BidIterator myBidIterator = new
				// BidIterator(this.utilitySpace.getDomain());
				if (count < 2) {
					for (double temputil = this.AvgUtilitythreshold; temputil <= this.MaximumUtility; temputil += 0.01) {
						// Bid tempbid = myBidIterator.next();
						// double temputil =
						// this.utilitySpace.getUtility(tempbid);
						double pchoose1 = (1 / this.relstdevUtility_Agent1
								* (1 / Math.sqrt(2 * Math.PI)))
								* Math.exp(-Math.pow(
										temputil - this.relavgUtility_Agent1, 2)
										/ (2 * Math.pow(
												this.relstdevUtility_Agent1,
												2)));
						double pchoose2 = (1 / this.relstdevUtility_Agent2
								* (1 / Math.sqrt(2 * Math.PI)))
								* Math.exp(-Math.pow(
										temputil - this.relavgUtility_Agent2, 2)
										/ (2 * Math.pow(
												this.relstdevUtility_Agent2,
												2)));
						double pchoose = (pchoose1 + pchoose2) / 2;
						if (pchoose >= Math.random()/*
													 * && temputil >=
													 * this.utilitythreshold
													 */) {
							return temputil;
						}
					}
				} else {
					return this.AvgUtilitythreshold;
				}
				count++;
			}

		} catch (Exception e) {
			System.out
					.println(e.getMessage() + "exception in method BidToOffer");
			this.except = 4;
		}
		// System.out.println("the current threshold is " +
		// this.utilitythreshold + " with the value of alpha1 is " + alpha1);
		return 1.0;
	}

	/*
	 * decide whether to accept the current offer or not
	 */
	private boolean AcceptOpponentOffer(Bid opponentBid, Bid ownBid) {
		try {
			double currentUtility = 0;
			double nextRoundUtility = 0;
			double maximumUtility = 0;
			this.concedeToOpponent = false;
			currentUtility = this.utilitySpace.getUtility(opponentBid);
			maximumUtility = this.MaximumUtility;// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);

			// System.out.println(this.utilitythreshold +"at time "+
			// timeline.getTime());
			if (currentUtility >= this.AvgUtilitythreshold
					|| currentUtility >= nextRoundUtility) {
				return true;
			} else {
				// if the current utility with discount is larger than the
				// predicted maximum utility with discount
				// then accept it.
				double predictMaximumUtility = maximumUtility
						* this.discountingFactor;
				// double currentMaximumUtility =
				// this.utilitySpace.getUtilityWithDiscount(opponentBidHistory.chooseBestFromHistory(utilitySpace),
				// timeline);
				double currentMaximumUtility1 = this.utilitySpace
						.getUtilityWithDiscount(
								opponentBidHistory1.getBestBidInHistory(),
								timeline);
				double currentMaximumUtility2 = this.utilitySpace
						.getUtilityWithDiscount(
								opponentBidHistory2.getBestBidInHistory(),
								timeline);
				double currentMaximumUtility = Math.min(currentMaximumUtility1,
						currentMaximumUtility2);
				if (currentMaximumUtility > predictMaximumUtility
						&& timeline.getTime() > this.AvgConcedeTime) {
					try {
						// if the current offer is approximately as good as the
						// best one in the history, then accept it.
						if (utilitySpace.getUtilityWithDiscount(opponentBid,
								timeline) >= currentMaximumUtility - 0.01) {
							System.out.println("he offered me "
									+ currentMaximumUtility
									+ " we predict we can get at most "
									+ predictMaximumUtility
									+ "we concede now to avoid lower payoff due to conflict");
							return true;
						} else {
							this.concedeToOpponent = true;
							return false;
						}
					} catch (Exception e) {
						System.out.println(
								"exception in Method AcceptOpponentOffer");
						this.except = 7;
						return true;
					}
					// retrieve the opponent's biding history and utilize it
				} else if (currentMaximumUtility > this.AvgUtilitythreshold
						* Math.pow(this.discountingFactor,
								timeline.getTime())) {
					try {
						// if the current offer is approximately as good as the
						// best one in the history, then accept it.
						if (utilitySpace.getUtilityWithDiscount(opponentBid,
								timeline) >= currentMaximumUtility - 0.01) {
							return true;
						} else {
							System.out.println("test"
									+ utilitySpace.getUtility(opponentBid)
									+ this.AvgUtilitythreshold);
							this.concedeToOpponent = true;
							return false;
						}
					} catch (Exception e) {
						System.out.println(
								"exception in Method AcceptOpponentOffer");
						this.except = 8;
						return true;
					}
				} else {
					return false;
				}
			}
		} catch (Exception e) {
			System.out.println("exception in Method AcceptOpponentOffer");
			return true;
		}

	}

	/*
	 * decide whether or not to terminate now
	 */
	private boolean TerminateCurrentNegotiation(Bid ownBid) {
		double currentUtility = 0;
		double nextRoundUtility = 0;
		double maximumUtility = 0;
		this.concedeToOpponent = false;
		try {
			currentUtility = this.reservationValue;
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
			maximumUtility = this.MaximumUtility;

			if (this.discountingFactor == 1 || this.reservationValue == 0) {
				return false;
			}

			if (currentUtility >= nextRoundUtility) {
				return true;
			} else {
				// if the current reseravation utility with discount is larger
				// than the predicted maximum utility with discount
				// then terminate the negotiation.
				double predictMaximumUtility = maximumUtility
						* this.discountingFactor;
				double currentMaximumUtility = this.utilitySpace
						.getReservationValueWithDiscount(timeline);
				// System.out.println("the current reserved value is "+
				// this.reservationValue+" after discounting is
				// "+currentMaximumUtility);
				if (currentMaximumUtility > predictMaximumUtility
						&& timeline.getTime() > this.AvgConcedeTime) {
					return true;
				} else {
					return false;
				}
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "Exception in method TerminateCurrentNegotiation");
			this.except = 9;
			return true;
		}
	}

	/*
	 * estimate the number of rounds left before reaching the deadline @param
	 * opponent @return
	 */
	private int estimateRoundLeft(boolean opponent) {
		double round;
		try {
			if (opponent == true) {
				if (this.timeLeftBefore
						- this.timeLeftAfter > this.maximumTimeOfOpponent) {
					this.maximumTimeOfOpponent = this.timeLeftBefore
							- this.timeLeftAfter;
				}
			} else {
				if (this.timeLeftAfter
						- this.timeLeftBefore > this.maximumTimeOfOwn) {
					this.maximumTimeOfOwn = this.timeLeftAfter
							- this.timeLeftBefore;
				}
			}
			if (this.maximumTimeOfOpponent + this.maximumTimeOfOwn == 0) {
				System.out.println("divided by zero exception");

				this.except = 10;
			}
			round = (this.totalTime - timeline.getCurrentTime())
					/ (this.maximumTimeOfOpponent + this.maximumTimeOfOwn);
			// System.out.println("current time is " +
			// timeline.getElapsedSeconds() + "---" + round + "----" +
			// this.maximumTimeOfOpponent);
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "Exception in method TerminateCurrentNegotiation");
			this.except = 9;
			return 20;
		}
		return ((int) (round));
	}

	/*
	 * pre-processing to save the computational time each round
	 */
	private void calculateBidsBetweenUtility() {
		BidIterator myBidIterator = null;
		myBidIterator = new BidIterator(this.utilitySpace.getDomain());
		int counts[] = new int[this.NumberofDivisions];
		try {
			this.MinimumUtility = Math.min(this.MinimumUtility_Agent1,
					this.MinimumUtility_Agent2);
			this.SpacebetweenDivisions = (this.MaximumUtility
					- this.MinimumUtility) / this.NumberofDivisions;
			// initalization for each arraylist storing the bids between each
			// range
			for (int i = 0; i < this.NumberofDivisions; i++) {
				ArrayList<Bid> BidList = null;
				BidList = new ArrayList<Bid>();
				// BidList.add(this.bid_maximum_utility);
				this.bidsBetweenUtility.add(i, BidList);
			}
			this.bidsBetweenUtility.get(this.NumberofDivisions - 1).add(
					counts[this.NumberofDivisions - 1],
					this.bid_maximum_utility);
			// note that here we may need to use some trick to reduce the
			// computation cost (to be checked later);
			// add those bids in each range into the corresponding arraylist
			while (myBidIterator.hasNext()) {
				Bid b = myBidIterator.next();
				for (int i = 0; i < this.NumberofDivisions; i++) {
					if (this.utilitySpace.getUtility(
							b) <= (i + 1) * this.SpacebetweenDivisions
									+ this.MinimumUtility
							&& this.utilitySpace.getUtility(b) >= i
									* this.SpacebetweenDivisions
									+ Math.min(this.MinimumUtility_Agent1,
											this.MinimumUtility_Agent2)) {
						this.bidsBetweenUtility.get(i).add(counts[i], b);
						counts[i]++;
						break;
					}
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in calculateBidsBetweenUtility()");
			this.except = 11;
		}
	}

	private Bid RandomSearchBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Bid bid = null;

		for (Issue lIssue : issues) {
			Random random = new Random();
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				int optionIndex = random
						.nextInt(lIssueDiscrete.getNumberOfValues());
				values.put(lIssue.getNumber(),
						lIssueDiscrete.getValue(optionIndex));
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				int optionInd = random.nextInt(
						lIssueReal.getNumberOfDiscretizationSteps() - 1);
				values.put(lIssueReal.getNumber(),
						new ValueReal(lIssueReal.getLowerBound() + (lIssueReal
								.getUpperBound() - lIssueReal.getLowerBound())
								* (optionInd) / (lIssueReal
										.getNumberOfDiscretizationSteps())));
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				int optionIndex2 = lIssueInteger.getLowerBound()
						+ random.nextInt(lIssueInteger.getUpperBound()
								- lIssueInteger.getLowerBound());
				values.put(lIssueInteger.getNumber(),
						new ValueInteger(optionIndex2));
				break;
			default:
				this.except = 12;
				throw new Exception(
						"issue type " + lIssue.getType() + " not supported");
			}
		}
		bid = new Bid(utilitySpace.getDomain(), values);
		return bid;
	}

	/*
	 * Get all the bids within a given utility range.
	 */
	private List<Bid> getBidsBetweenUtility(double lowerBound,
			double upperBound) {
		List<Bid> bidsInRange = new ArrayList<Bid>();
		try {
			int range = (int) ((upperBound - this.minimumUtilityThreshold)
					/ 0.01);
			int initial = (int) ((lowerBound - this.minimumUtilityThreshold)
					/ 0.01);
			// System.out.println(range+"---"+initial);
			for (int i = initial; i < range; i++) {
				bidsInRange.addAll(i, this.bidsBetweenUtility.get(i));
			}
			if (bidsInRange.isEmpty()) {
				bidsInRange.add(range - 1, this.bid_maximum_utility);
			}
		} catch (Exception e) {
			System.out.println("Exception in getBidsBetweenUtility");
			this.except = 13;
			e.printStackTrace();
		}
		return bidsInRange;
	}

	/*
	 * determine concede-to-time degree based on the discounting factor.
	 */
	private void chooseConcedeToDiscountingDegree() {
		try {
			double alpha = 0;
			double beta = 1.5;// 1.3;//this value controls the rate at which the
								// agent concedes to the discouting factor.
			// the larger beta is, the more the agent makes concesions.
			// if (utilitySpace.getDomain().getNumberOfPossibleBids() > 100) {
			/*
			 * if (this.maximumOfBid > 100) { beta = 2;//1.3; } else { beta =
			 * 1.5; }
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
			this.AvgConcedeTime = this.minConcedeTime
					+ (1 - this.minConcedeTime) * alpha;
			this.concedeTime_original = this.AvgConcedeTime;
			// System.out.println("concedeToDiscountingFactor is " +
			// this.AvgConcedeTime + "current time is " + timeline.getTime());
		} catch (Exception e) {
			System.out.println(
					"Exception in method chooseConcedeToDiscountingDegree");
		}
	}

	/*
	 * update the concede-to-time degree based on the predicted toughness degree
	 * of the opponent
	 */

	private Bid regeneratebid(double bidutil) {
		Bid ans = null;
		ArrayList<Bid> Bids = null;
		this.mindistance = 999;
		this.averagedistance = 999;
		int count = 0;
		try {
			int check = 0;
			if (this.OppBids1.get(this.storingcapacity - 1) == null
					|| this.OppBids2.get(this.storingcapacity - 1) == null) {
				ans = this.bid_maximum_utility;
			} else {
				for (this.CurrentrangeNumber = 0; check == 0; this.CurrentrangeNumber++) {
					if (bidutil < ((this.CurrentrangeNumber + 1)
							* this.SpacebetweenDivisions
							+ this.MinimumUtility)) {
						check = 1;
						break;
					}
				}
				Bids = this.bidsBetweenUtility.get(this.CurrentrangeNumber);
				while (Bids.size() <= 10) {
					if (this.CurrentrangeNumber <= this.NumberofDivisions - 2) {
						for (int i = 0; i < this.bidsBetweenUtility
								.get(this.CurrentrangeNumber + 1).size(); i++) {
							Bids.add(Bids.size(), this.bidsBetweenUtility
									.get(this.CurrentrangeNumber + 1).get(i));
						}
					} else {
						break;
					}
					this.CurrentrangeNumber++;
				}
				this.size = Bids.size();
				if (Bids == null) {
					ans = this.mylastBid;
				} else {
					if (Math.random() > 0.1) {
						for (Bid Bid1 : Bids) {
							if (this.utilitySpace.getUtility(
									Bid1) > this.AvgUtilitythreshold) {
								count = 0;
								for (int i = 0; i < this.storingcapacity; i++) {
									if (this.OppBids1.get(i) != null) {
										this.averagedistance += calculatedistance(
												Bid1, this.OppBids1.get(i));
										count++;
									}
									if (this.OppBids2.get(i) != null) {
										this.averagedistance += calculatedistance(
												Bid1, this.OppBids2.get(i));
										count++;
									}
								}
								this.averagedistance /= count;
								if (this.averagedistance < this.mindistance) {
									this.mindistance = this.averagedistance;
									ans = Bid1;
								} else if (this.averagedistance == this.mindistance
										&& Math.random() <= 0.5) {
									this.mindistance = this.averagedistance;
									ans = Bid1;
								}
							}
						}
					} else {
						ans = Bids.get((int) (Math.random() * Bids.size()));
					}
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in method regeneratebid");
			this.except = 17;
			ans = this.bid_maximum_utility;
		}
		return ans;

	}

	private HashMap<Issue, Value> getBidValues(Bid bid) {
		try {
			HashMap<Issue, Value> Values = new HashMap<Issue, Value>();
			for (Issue lIssue : this.Issues) {
				int issueNum = lIssue.getNumber();
				Value v = bid.getValue(issueNum);
				Values.put(lIssue, v);
			}
			return Values;
		} catch (Exception e) {
			System.out
					.println("Exception in getBidValues(): " + e.getMessage());
			this.except = 18;
		}
		return null;
	}

	private double calculatedistance(Bid Bid1, Bid Bid2) {
		try {
			double distance = 0.0D;
			HashMap<Issue, Value> Map1 = this.getBidValues(Bid1);
			HashMap<Issue, Value> Map2 = this.getBidValues(Bid2);
			for (Issue Issue1 : this.Issues) {
				switch (Issue1.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) Issue1;
					ValueDiscrete value1 = (ValueDiscrete) Map1.get(Issue1);
					ValueDiscrete value2 = (ValueDiscrete) Map2.get(Issue1);
					int index1 = lIssueDiscrete.getValueIndex(value1);
					int index2 = lIssueDiscrete.getValueIndex(value2);
					distance += Math.abs(index1 - index2);

					break;
				case REAL:
					ValueReal valueReal1 = (ValueReal) Map1.get(Issue1);
					ValueReal valueReal2 = (ValueReal) Map2.get(Issue1);
					double indexReal1 = valueReal1.getValue();
					double indexReal2 = valueReal2.getValue();
					distance += Math.abs(indexReal1 - indexReal2);
					break;
				case INTEGER:
					ValueInteger valueInteger1 = (ValueInteger) Map1
							.get(Issue1);
					ValueInteger valueInteger2 = (ValueInteger) Map2
							.get(Issue1);
					double indexInteger1 = valueInteger1.getValue();
					double indexInteger2 = valueInteger2.getValue();
					distance += Math.abs(indexInteger1 - indexInteger2);
					break;
				}
			}
			return distance;
		} catch (Exception e) {
			System.out.println("Exception in method calculatedistance");
			this.except = 19;
			return 999;
		}
	}

	private void updateConcedeDegree_smallDiscountingFactor() {
		double gama = 10;
		double weight = 0.1;
		double opponnetToughnessDegree = this.opponentBidHistory2
				.getConcessionDegree();
		try {
			if (IsOppFirstBid_Agent1) {
				this.oppFirstBidUtility_Agent1 = utilitySpace
						.getUtility(this.oppFirstBid_Agent1);
				if (this.reservationValue <= this.oppFirstBidUtility_Agent1) {
					this.MinimumUtility_Agent1 = this.oppFirstBidUtility_Agent1;
				}
				IsOppFirstBid_Agent1 = false;
			}

			if (IsOppFirstBid_Agent2) {
				this.oppFirstBidUtility_Agent2 = utilitySpace
						.getUtility(this.oppFirstBid_Agent2);
				if (this.reservationValue <= this.oppFirstBidUtility_Agent2) {
					this.MinimumUtility_Agent2 = this.oppFirstBidUtility_Agent2;
				}
				IsOppFirstBid_Agent2 = false;
			}

			double temp = this.concedeTime_original
					+ weight * (1 - this.concedeTime_original)
							* Math.pow(opponnetToughnessDegree, gama);
			this.concedeTime_Agent1 = temp;
			this.concedeTime_Agent2 = temp;
			if ((this.concedeTime_Agent1 >= 1)
					|| (this.concedeTime_Agent2 >= 1)) {
				this.concedeTime_Agent1 = 1;
				this.concedeTime_Agent2 = 1;
			}
		} catch (Exception e) {
			System.out.println(
					"Exception in method updateConcedeDegree_smallDiscountingFactor");
			this.except = 20;
		}
		// System.out.println("concedeToDiscountingFactor is " +
		// this.concedeToDiscountingFactor + "current time is " +
		// timeline.getTime() + "original concedetodiscoutingfactor is " +
		// this.concedeToDiscountingFactor_original);
	}

	private void updateConcedeDegree_largeDiscountingFactor_Agent1() {
		try {
			double i = 0;
			if (IsOppFirstBid_Agent1) {
				this.CalculateMinThreshold_Agent1();
				IsOppFirstBid_Agent1 = false;
			}

			this.MeasureConcedePartOfOppBehaviour_Agent1();

			this.concedePartOfdiscountingFactor_Agent1 = this.discountingFactor
					- 1;
			this.concedePartOfOppBehaviour_Agent1 = (((((this.relcountLowerBoundMid2_Agent1
					/ this.relcountUpperBoundMid2_Agent1)
					* this.slopeOfSlopeOfSessionMax2_Agent1)
					- ((this.relcountLowerBoundMid1_Agent1
							/ this.relcountUpperBoundMid1_Agent1)
							* this.slopeOfSlopeOfSessionMax1_Agent1))
					/ this.k) / this.relstdevUtility_Agent1) - this.N;
			if (this.startConcede_Agent1 == true) {
				i = this.concedePartOfdiscountingFactor_Agent1
						+ this.concedePartOfOppBehaviour_Agent1;
				this.concedeTime_Agent1 = Math.exp(i);
				if (this.concedeTime_Agent1 > 1) {
					this.concedeTime_Agent1 = 1;
				}
			} else {
				this.concedeTime_Agent2 = 0.5;
			}
		} catch (Exception e) {
			this.except = 23;
			System.out.println(
					"updateConcedeDegree_largeDiscountingFactor_Agent1 exception");
		}
	}

	private void updateConcedeDegree_largeDiscountingFactor_Agent2() {
		try {
			double i = 0;
			if (IsOppFirstBid_Agent2) {
				this.CalculateMinThreshold_Agent2();
				IsOppFirstBid_Agent2 = false;
			}

			this.MeasureConcedePartOfOppBehaviour_Agent2();

			this.concedePartOfdiscountingFactor_Agent2 = this.discountingFactor
					- 1;
			this.concedePartOfOppBehaviour_Agent2 = (((((this.relcountLowerBoundMid2_Agent2
					/ this.relcountUpperBoundMid2_Agent2)
					* this.slopeOfSlopeOfSessionMax2_Agent2)
					- ((this.relcountLowerBoundMid1_Agent2
							/ this.relcountUpperBoundMid1_Agent2)
							* this.slopeOfSlopeOfSessionMax1_Agent2))
					/ this.k) / this.relstdevUtility_Agent2) - this.N;
			if (this.startConcede_Agent2 == true) {
				i = this.concedePartOfdiscountingFactor_Agent2
						+ this.concedePartOfOppBehaviour_Agent2;
				this.concedeTime_Agent2 = Math.exp(i);
				if (this.concedeTime_Agent2 > 1) {
					this.concedeTime_Agent2 = 1;
				}
			} else {
				this.concedeTime_Agent2 = 0.5;
			}
		} catch (Exception e) {
			this.except = 23;
			System.out.println(
					"updateConcedeDegree_largeDiscountingFactor_Agent2 exception");
		}
	}

	private void updateConcedeDegree_nonDiscountingFactor_Agent1() {
		try {
			if (IsOppFirstBid_Agent1) {
				this.CalculateMinThreshold_Agent1();
				IsOppFirstBid_Agent1 = false;
			}

			this.concedeTime_Agent1 = 1;
		} catch (Exception e) {
			this.except = 25;
			System.out.println(
					"updateConcedeDegree_nonDiscountingFactor_Agent1 exception");
		}
	}

	private void updateConcedeDegree_nonDiscountingFactor_Agent2() {
		try {
			if (IsOppFirstBid_Agent2) {
				this.CalculateMinThreshold_Agent2();
				IsOppFirstBid_Agent2 = false;
			}

			this.concedeTime_Agent2 = 1;
		} catch (Exception e) {
			this.except = 25;
			System.out.println(
					"updateConcedeDegree_nonDiscountingFactor_Agent2 exception");
		}
	}

	private void CalculateMinThreshold_Agent1() {
		try {
			ArrayList<Double> bidsUtil = new ArrayList();
			bidsUtil = this.CalculateSumAndCount_Agent1(bidsUtil);

			this.relavgUtility_Agent1 = this.relsumUtility_Agent1
					/ this.relcountUtility_Agent1;

			for (Iterator f = bidsUtil.iterator(); f.hasNext();) {
				double bidUtil = (Double) f.next();

				if (bidUtil >= this.oppFirstBidUtility_Agent1) {
					this.relSqRootOfAvgUtility_Agent1 += (bidUtil
							- this.relavgUtility_Agent1)
							* (bidUtil - this.relavgUtility_Agent1);
				}
			}
			this.relstdevUtility_Agent1 = Math
					.sqrt(this.relSqRootOfAvgUtility_Agent1
							/ this.relcountUtility_Agent1);

			if (this.relstdevUtility_Agent1 < 0.1) {
				this.relSqRootOfAvgUtility_Agent1 = 0;
				for (Iterator g = bidsUtil.iterator(); g.hasNext();) {
					double bidUtil = (Double) g.next();

					if (bidUtil >= this.oppFirstBidUtility_Agent1
							&& (bidUtil < (this.relavgUtility_Agent1
									- this.relstdevUtility_Agent1)
									|| bidUtil > (this.relavgUtility_Agent1
											+ this.relstdevUtility_Agent1))) {
						this.relSqRootOfAvgUtility_Agent1 += (bidUtil
								- this.relavgUtility_Agent1)
								* (bidUtil - this.relavgUtility_Agent1);
						this.relcountUtilityInSmallSD_Agent1++;
					}
				}
				this.relstdevUtility_Agent1 = Math
						.sqrt(this.relSqRootOfAvgUtility_Agent1
								/ this.relcountUtilityInSmallSD_Agent1);
			}

			this.minThreshold_Agent1 = this.relavgUtility_Agent1
					+ this.discountingFactor * this.relstdevUtility_Agent1
							* this.reservationValue;

			if (this.relcountUtility_Agent1 < 20) {
				this.minThreshold_Agent1 = this.relavgUtility_Agent1;
			}

			this.Compare_MinThreshold_And_SecondMax_Agent1();
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method CalculateMinThreshold_Agent1");
		}
	}

	private void CalculateMinThreshold_Agent2() {
		try {
			ArrayList<Double> bidsUtil = new ArrayList();
			bidsUtil = this.CalculateSumAndCount_Agent2(bidsUtil);

			this.relavgUtility_Agent2 = this.relsumUtility_Agent2
					/ this.relcountUtility_Agent2;

			for (Iterator f = bidsUtil.iterator(); f.hasNext();) {
				double bidUtil = (Double) f.next();

				if (bidUtil >= this.oppFirstBidUtility_Agent2) {
					this.relSqRootOfAvgUtility_Agent2 += (bidUtil
							- this.relavgUtility_Agent2)
							* (bidUtil - this.relavgUtility_Agent2);
				}
			}
			this.relstdevUtility_Agent2 = Math
					.sqrt(this.relSqRootOfAvgUtility_Agent2
							/ this.relcountUtility_Agent2);

			if (this.relstdevUtility_Agent2 < 0.1) {
				this.relSqRootOfAvgUtility_Agent2 = 0;
				for (Iterator g = bidsUtil.iterator(); g.hasNext();) {
					double bidUtil = (Double) g.next();

					if (bidUtil >= this.oppFirstBidUtility_Agent2
							&& (bidUtil < (this.relavgUtility_Agent2
									- this.relstdevUtility_Agent2)
									|| bidUtil > (this.relavgUtility_Agent2
											+ this.relstdevUtility_Agent2))) {
						this.relSqRootOfAvgUtility_Agent2 += (bidUtil
								- this.relavgUtility_Agent2)
								* (bidUtil - this.relavgUtility_Agent2);
						this.relcountUtilityInSmallSD_Agent2++;
					}
				}
				this.relstdevUtility_Agent2 = Math
						.sqrt(this.relSqRootOfAvgUtility_Agent2
								/ this.relcountUtilityInSmallSD_Agent2);
			}

			this.minThreshold_Agent2 = this.relavgUtility_Agent2
					+ this.discountingFactor * this.relstdevUtility_Agent2
							* this.reservationValue;

			if (this.relcountUtility_Agent2 < 20) {
				this.minThreshold_Agent2 = this.relavgUtility_Agent2;
			}

			this.Compare_MinThreshold_And_SecondMax_Agent2();
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method CalculateMinThreshold_Agent2");
		}
	}

	private void Compare_MinThreshold_And_SecondMax_Agent1() {
		if (this.minThreshold_Agent1 > utility_SecondMaximum_Agent1) {
			this.minThreshold_Agent1 = utility_SecondMaximum_Agent1 * 0.9;
		}
	}

	private void Compare_MinThreshold_And_SecondMax_Agent2() {
		if (this.minThreshold_Agent2 > utility_SecondMaximum_Agent2) {
			this.minThreshold_Agent2 = utility_SecondMaximum_Agent2 * 0.9;
		}
	}

	private ArrayList<Double> CalculateSumAndCount_Agent1(
			ArrayList<Double> bidsUtil) {
		try {
			this.oppFirstBidUtility_Agent1 = utilitySpace
					.getUtility(this.oppFirstBid_Agent1);
			if (this.reservationValue <= this.oppFirstBidUtility_Agent1) {
				this.MinimumUtility_Agent1 = this.oppFirstBidUtility_Agent1;
			}

			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			for (; myBidIterator.hasNext();) {
				try {
					bidsUtil.add(
							this.utilitySpace.getUtility(myBidIterator.next()));
				} catch (Exception e) {
					System.out.println(e.getMessage()
							+ "exception in method CalculateSumAndCount_Agent1");
					return null;
				}
			}
			this.relavgUtility_Agent1 = 0;
			this.relsumUtility_Agent1 = 0;
			this.relcountUtility_Agent1 = 0;
			this.relSqRootOfAvgUtility_Agent1 = 0;
			this.relstdevUtility_Agent1 = 0;
			this.minThreshold_Agent1 = 0;

			for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
				double bidUtil = (Double) e.next();
				if (this.utility_FirstMaximum_Agent1 < bidUtil) {
					this.utility_FirstMaximum_Agent1 = bidUtil;
				} else if (utility_SecondMaximum_Agent1 < bidUtil) {
					this.utility_SecondMaximum_Agent1 = bidUtil;
				}

				if (bidUtil >= this.oppFirstBidUtility_Agent1) {
					this.relsumUtility_Agent1 += bidUtil;
					this.relcountUtility_Agent1 += 1;
				}
			}

			return bidsUtil;
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method CalculateSumAndCount_Agent1");
		}

		return null;
	}

	private ArrayList<Double> CalculateSumAndCount_Agent2(
			ArrayList<Double> bidsUtil) {
		try {
			this.oppFirstBidUtility_Agent2 = utilitySpace
					.getUtility(this.oppFirstBid_Agent2);
			if (this.reservationValue <= this.oppFirstBidUtility_Agent2) {
				this.MinimumUtility_Agent2 = this.oppFirstBidUtility_Agent2;
			}

			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			for (; myBidIterator.hasNext();) {
				bidsUtil.add(
						this.utilitySpace.getUtility(myBidIterator.next()));
			}
			this.relavgUtility_Agent2 = 0;
			this.relsumUtility_Agent2 = 0;
			this.relcountUtility_Agent2 = 0;
			this.relSqRootOfAvgUtility_Agent2 = 0;
			this.relstdevUtility_Agent2 = 0;
			this.minThreshold_Agent2 = 0;

			for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
				double bidUtil = (Double) e.next();
				if (this.utility_FirstMaximum_Agent2 < bidUtil) {
					this.utility_FirstMaximum_Agent2 = bidUtil;
				} else if (utility_SecondMaximum_Agent2 < bidUtil) {
					this.utility_SecondMaximum_Agent2 = bidUtil;
				}

				if (bidUtil >= this.oppFirstBidUtility_Agent2) {
					this.relsumUtility_Agent2 += bidUtil;
					this.relcountUtility_Agent2 += 1;
				}
			}

			return bidsUtil;
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method CalculateSumAndCount_Agent2");
			return null;
		}
	}

	private void MeasureConcedePartOfOppBehaviour_Agent1() {
		try {
			count_Agent1++;

			if (count_Agent1 < this.FirstTimeInterval) {
				if (utilitySpace.getUtility(
						this.oppPreviousBid_Agent1) > this.utility_maximum_from_opponent_Session1_Agent1) {
					this.utility_maximum_from_opponent_Session1_Agent1 = utilitySpace
							.getUtility(this.oppPreviousBid_Agent1);
				}
			} else if (count_Agent2 < this.SecondTimeInterval) {
				if (utilitySpace.getUtility(
						this.oppPreviousBid_Agent1) > this.utility_maximum_from_opponent_Session2_Agent1) {
					this.utility_maximum_from_opponent_Session2_Agent1 = utilitySpace
							.getUtility(this.oppPreviousBid_Agent1);
				}
			} else if (count_Agent2 < this.ThirdTimeInterval) {
				if (utilitySpace.getUtility(
						this.oppPreviousBid_Agent1) > this.utility_maximum_from_opponent_Session3_Agent1) {
					this.utility_maximum_from_opponent_Session3_Agent1 = utilitySpace
							.getUtility(this.oppPreviousBid_Agent1);
				}
			} else {
				this.relcountUpperBoundMid1_Agent1 = 0;
				this.relcountUpperBoundMid2_Agent1 = 0;
				this.relcountLowerBoundMid1_Agent1 = 0;
				this.relcountLowerBoundMid2_Agent1 = 0;
				this.midPointOfSlopeSessionMax1_Agent1 = (this.utility_maximum_from_opponent_Session2_Agent1
						+ this.utility_maximum_from_opponent_Session1_Agent1)
						/ 2;
				this.midPointOfSlopeSessionMax2_Agent1 = (this.utility_maximum_from_opponent_Session3_Agent1
						+ this.utility_maximum_from_opponent_Session2_Agent1)
						/ 2;
				this.slopeOfSlopeOfSessionMax1_Agent1 = this.utility_maximum_from_opponent_Session2_Agent1
						- this.utility_maximum_from_opponent_Session1_Agent1;
				this.slopeOfSlopeOfSessionMax2_Agent1 = this.utility_maximum_from_opponent_Session3_Agent1
						- this.utility_maximum_from_opponent_Session2_Agent1;

				ArrayList<Double> bidsUtil = new ArrayList();
				BidIterator myBidIterator = new BidIterator(
						this.utilitySpace.getDomain());
				for (; myBidIterator.hasNext();) {
					bidsUtil.add(
							this.utilitySpace.getUtility(myBidIterator.next()));
				}
				for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
					double bidUtil = (Double) e.next();
					if (bidUtil >= this.midPointOfSlopeSessionMax1_Agent1) {
						this.relcountUpperBoundMid1_Agent1 += 1;
					} else {
						this.relcountLowerBoundMid1_Agent1 += 1;
					}

					if (bidUtil >= this.midPointOfSlopeSessionMax2_Agent1) {
						this.relcountUpperBoundMid2_Agent1 += 1;
					} else {
						this.relcountLowerBoundMid2_Agent1 += 1;
					}
				}
				this.utility_maximum_from_opponent_Session1_Agent1 = this.utility_maximum_from_opponent_Session2_Agent1;
				this.utility_maximum_from_opponent_Session2_Agent1 = this.utility_maximum_from_opponent_Session3_Agent1;
				this.utility_maximum_from_opponent_Session3_Agent1 = 0;
				this.count_Agent1 = this.SecondTimeInterval - 1;
				this.startConcede_Agent1 = true;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method MeasureConcedePartOfOppBehaviour_Agent1");
		}
	}

	private void MeasureConcedePartOfOppBehaviour_Agent2() {
		try {
			count_Agent2++;

			if (count_Agent2 < this.FirstTimeInterval) {
				if (utilitySpace.getUtility(
						this.oppPreviousBid_Agent2) > this.utility_maximum_from_opponent_Session1_Agent2) {
					this.utility_maximum_from_opponent_Session1_Agent2 = utilitySpace
							.getUtility(this.oppPreviousBid_Agent2);
				}
			} else if (count_Agent2 < this.SecondTimeInterval) {
				if (utilitySpace.getUtility(
						this.oppPreviousBid_Agent2) > this.utility_maximum_from_opponent_Session2_Agent2) {
					this.utility_maximum_from_opponent_Session2_Agent2 = utilitySpace
							.getUtility(this.oppPreviousBid_Agent2);
				}
			} else if (count_Agent2 < this.ThirdTimeInterval) {
				if (utilitySpace.getUtility(
						this.oppPreviousBid_Agent2) > this.utility_maximum_from_opponent_Session3_Agent2) {
					this.utility_maximum_from_opponent_Session3_Agent2 = utilitySpace
							.getUtility(this.oppPreviousBid_Agent2);
				}
			} else {
				this.relcountUpperBoundMid1_Agent2 = 0;
				this.relcountUpperBoundMid2_Agent2 = 0;
				this.relcountLowerBoundMid1_Agent2 = 0;
				this.relcountLowerBoundMid2_Agent2 = 0;
				this.midPointOfSlopeSessionMax1_Agent2 = (this.utility_maximum_from_opponent_Session2_Agent2
						+ this.utility_maximum_from_opponent_Session1_Agent2)
						/ 2;
				this.midPointOfSlopeSessionMax2_Agent2 = (this.utility_maximum_from_opponent_Session3_Agent2
						+ this.utility_maximum_from_opponent_Session2_Agent2)
						/ 2;
				this.slopeOfSlopeOfSessionMax1_Agent2 = this.utility_maximum_from_opponent_Session2_Agent2
						- this.utility_maximum_from_opponent_Session1_Agent2;
				this.slopeOfSlopeOfSessionMax2_Agent2 = this.utility_maximum_from_opponent_Session3_Agent2
						- this.utility_maximum_from_opponent_Session2_Agent2;

				ArrayList<Double> bidsUtil = new ArrayList();
				BidIterator myBidIterator = new BidIterator(
						this.utilitySpace.getDomain());
				for (; myBidIterator.hasNext();) {
					bidsUtil.add(
							this.utilitySpace.getUtility(myBidIterator.next()));
				}
				for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
					double bidUtil = (Double) e.next();
					if (bidUtil >= this.midPointOfSlopeSessionMax1_Agent2) {
						this.relcountUpperBoundMid1_Agent2 += 1;
					} else {
						this.relcountLowerBoundMid1_Agent2 += 1;
					}

					if (bidUtil >= this.midPointOfSlopeSessionMax2_Agent2) {
						this.relcountUpperBoundMid2_Agent2 += 1;
					} else {
						this.relcountLowerBoundMid2_Agent2 += 1;
					}
				}
				this.utility_maximum_from_opponent_Session1_Agent2 = this.utility_maximum_from_opponent_Session2_Agent2;
				this.utility_maximum_from_opponent_Session2_Agent2 = this.utility_maximum_from_opponent_Session3_Agent2;
				this.utility_maximum_from_opponent_Session3_Agent2 = 0;
				this.count_Agent2 = SecondTimeInterval - 1;
				this.startConcede_Agent2 = true;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method MeasureConcedePartOfOppBehaviour_Agent2");
		}
	}

	private Bid BidToOffer_original() {
		Bid bidReturned = null;
		int test = 0;

		try {
			double maximumOfBid = this.MaximumUtility;// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			double minimumOfBid;

			if (timeline.getTime() <= ((this.concedeTime_Agent2
					+ this.concedeTime_Agent1) / 2)) {
				if (this.discountingFactor <= 0.5) {
					this.minThreshold_Agent1 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.concedeTime_Agent1);
					this.minThreshold_Agent2 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.concedeTime_Agent2);
					this.utilitythreshold_Agent1 = maximumOfBid
							- (maximumOfBid - this.minThreshold_Agent1)
									* Math.pow(
											(timeline.getTime()
													/ this.concedeTime_Agent1),
											alpha1);
					this.utilitythreshold_Agent2 = maximumOfBid
							- (maximumOfBid - this.minThreshold_Agent2)
									* Math.pow(
											(timeline.getTime()
													/ this.concedeTime_Agent2),
											alpha1);
				} else {
					this.utilitythreshold_Agent1 = this.minThreshold_Agent1
							+ (maximumOfBid - this.minThreshold_Agent1) * (1
									- Math.sin((Math.PI / 2) * (timeline
											.getTime() / this.concedeTime_Agent1)));
					this.utilitythreshold_Agent2 = this.minThreshold_Agent2
							+ (maximumOfBid - this.minThreshold_Agent2) * (1
									- Math.sin((Math.PI / 2) * (timeline
											.getTime() / this.concedeTime_Agent2)));
				}
			} else {
				if (this.discountingFactor <= 0.5) {
					this.utilitythreshold_Agent1 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									timeline.getTime());
					this.utilitythreshold_Agent2 = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									timeline.getTime());
				} else {
					this.utilitythreshold_Agent1 = this.minThreshold_Agent1
							+ (maximumOfBid - this.minThreshold_Agent1)
									/ (1 - concedeTime_Agent1)
									* Math.pow(
											(timeline.getTime()
													- this.concedeTime_Agent1),
											this.discountingFactor);
					this.utilitythreshold_Agent2 = this.minThreshold_Agent2
							+ (maximumOfBid - this.minThreshold_Agent2)
									/ (1 - concedeTime_Agent2)
									* Math.pow(
											(timeline.getTime()
													- this.concedeTime_Agent2),
											this.discountingFactor);
				}
			}
			this.AvgUtilitythreshold = (this.utilitythreshold_Agent2
					+ this.utilitythreshold_Agent1) / 2;
			if (this.AvgUtilitythreshold > MaximumUtility) {
				this.AvgUtilitythreshold = MaximumUtility;
			}

			minimumOfBid = this.AvgUtilitythreshold;

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
			Bid bestBidOfferedByOpponent1 = opponentBidHistory1
					.getBestBidInHistory();
			Bid bestBidOfferedByOpponent2 = opponentBidHistory2
					.getBestBidInHistory();
			Bid bestBidOfferedByOpponent = this
					.getUtility(bestBidOfferedByOpponent1) > this
							.getUtility(bestBidOfferedByOpponent2)
									? bestBidOfferedByOpponent2
									: bestBidOfferedByOpponent1;
			if (utilitySpace.getUtility(
					bestBidOfferedByOpponent) >= this.AvgUtilitythreshold
					|| utilitySpace.getUtility(
							bestBidOfferedByOpponent) >= minimumOfBid) {
				return bestBidOfferedByOpponent;
			}
			this.nextbidutil = this.BidToOffer();
			bidReturned = this.regeneratebid(this.nextbidutil);
			if (bidReturned == null) {
				System.out.println("no bid is searched warning");
				bidReturned = this.utilitySpace.getMaxUtilityBid();
			}
			MinAcceptCondition(bidReturned);
		} catch (Exception e) {
			System.out.println(
					e.getMessage() + "exception in method BidToOffer_original");
			this.except = 26;
			return this.bid_maximum_utility;
		}
		// System.out.println("the current threshold is " +
		// this.utilitythreshold + " with the value of alpha1 is " + alpha1);
		return bidReturned;
	}

	private boolean OtherAcceptCondition(boolean IsAccept) {
		try {
			if ((timeline.getTime() >= ((this.concedeTime_Agent2
					+ this.concedeTime_Agent1) / 2))
					&& (this.utilitySpace.getUtility(
							this.oppPreviousBid_Agent2) >= this.minUtilityUhreshold)) {
				IsAccept = true;
			}
		} catch (Exception e) {
			this.except = 27;
			System.out.println(e.getMessage()
					+ "exception in method OtherAcceptCondition");
			return true;
		}
		return IsAccept;
	}

	private void MinAcceptCondition(Bid bidReturned) {
		try {
			if (this.minUtilityUhreshold > this.utilitySpace
					.getUtility(bidReturned)) {
				this.minUtilityUhreshold = this.utilitySpace
						.getUtility(bidReturned);
			}
		} catch (Exception e) {
			this.except = 28;
			System.out.println(
					e.getMessage() + "exception in method MinAcceptCondition");
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}
}
