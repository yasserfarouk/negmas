package agents.anac.y2016.ngent;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
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

/**
 *
 * @author Tom and Tommy
 * 
 *         Ngent: Bidding Method: create 3D Space of bid to find the Nash bid
 *         Only add the issue score using exponential function (Not normalize
 *         each issue score) For nonDF, concedeTime = 0.5 For smallDF, MinU will
 *         be calculated by CUHKAgent's function Turning point exists Using
 *         Max's function to adjust the adaptive minU Max num of bids: 7000 If
 *         domain size > 1000, use log freq. Otherwise, use freq
 * 
 */
public class Ngent extends AbstractNegotiationParty {
	boolean debug = false;

	private int round;
	private final double totalTime = 180;
	private OpponentBidHistory opponentBidHistory1;
	private OpponentBidHistory opponentBidHistory2;
	private double minimumUtilityThreshold;
	private double avgUtilitythreshold;
	private double avgConcedeTime;
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
	private LinkedList<LinkedList<Bid>> bidsBetweenUtility;
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

	private double initialConcedeTime;
	private double initialUtilitythreshold;
	/* Agent 1 Analysis */
	private AgentData agentData1;
	/* Agent 2 Analysis */
	private AgentData agentData2;

	private double MinimumUtility;
	private double adaptiveMinUThreshold;
	private double finalMinUAgent1 = 0;
	private double finalMinUAgent2 = 0;
	private double offeredBidsAvg;
	private double offeredBidsSD;
	private int offeredBidsRound;
	private AgentID partyId;
	private Action ActionOfOpponent = null;
	private int NumberofDivisions;
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
	private Bid myLastBid;

	private HashMap<Bid, Double> bidsUtilityScore;
	private int size;
	private double minUtilityUhreshold = 1.1;
	private int countOpp = 0;
	private int EachRoundCount = 0;
	private final int maxNumOfBids = 7000; // the maximum number of bids to be
											// considered in a round
	private int largeDomainSize; // the maximum number of bids to be considered
									// in a round
	private boolean isInProfile3;
	private boolean isNeededToMoveProfile3;
	private double domainSize;

	PrintStream file, SysOut;
	String fileName;
	String string_AddIssueScore;
	String string_SubtractIssueScore;
	String string_agentData1Num;
	String string_agentData2Num;

	private class AgentData {
		LinkedList<Double> issueScoreBids;
		ArrayList<ArrayList<Double>> issueScore;
		double utility_maximum_from_opponent_Session1;
		double utility_maximum_from_opponent_Session2;
		double utility_maximum_from_opponent_Session3;
		double count;
		double concedeTime;
		double oppFirstBidUtility;
		double concedePartOfdiscountingFactor;
		double concedePartOfOppBehaviour;
		double minThreshold;
		Bid oppFirstBid;
		boolean startConcede;
		double relCountUpperBoundMid1;
		double relCountLowerBoundMid1;
		double relCountUpperBoundMid2;
		double relCountLowerBoundMid2;
		double relStdevUtility;
		double utility_FirstMaximum;
		double utility_SecondMaximum;
		double midPointOfSlopeSessionMax1;
		double midPointOfSlopeSessionMax2;
		double slopeOfSlopeOfSessionMax1;
		double slopeOfSlopeOfSessionMax2;
		boolean IsOppFirstBid;
		Bid oppPreviousBid;
		double MinimumUtility;
		double utilitythreshold;
		String agentNum;
		double maxScore;
		double avgScore;
		boolean isReceivedOppFirstBid;

		private AgentData(double minimumUtility, double initialConcedeTime,
				double initialUtilitythreshold, String agentNumber) {
			utility_maximum_from_opponent_Session1 = 0;
			utility_maximum_from_opponent_Session2 = 0;
			utility_maximum_from_opponent_Session3 = 0;
			count = -1;
			startConcede = false;
			relStdevUtility = 1;
			utility_FirstMaximum = 0;
			utility_SecondMaximum = 0;
			midPointOfSlopeSessionMax1 = 0;
			midPointOfSlopeSessionMax2 = 0;
			slopeOfSlopeOfSessionMax1 = 0;
			slopeOfSlopeOfSessionMax2 = 0;
			MinimumUtility = minimumUtility;
			concedeTime = initialConcedeTime;
			oppFirstBidUtility = 0;
			oppFirstBid = null;
			IsOppFirstBid = true;
			isReceivedOppFirstBid = false;
			utilitythreshold = initialUtilitythreshold;
			minThreshold = 0;
			agentNum = agentNumber;
			issueScore = new ArrayList<ArrayList<Double>>();
		}

		private boolean GetIsReceivedOppFirstBid() {
			return isReceivedOppFirstBid;
		}

		private void SetIsReceivedOppFirstBid(boolean state) {
			isReceivedOppFirstBid = state;
		}

		private void ChangeIssueScore(Bid bid, String caseName) {
			try {
				double addedScore = 0;
				if (caseName.equals(string_AddIssueScore)) {
					addedScore = Math.exp(-timeline.getTime());
				} else {
					addedScore = -(1 - Math
							.exp(-timeline.getTime() / discountingFactor * 50));
				}
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("bid:" + this.utilitySpace.getUtility(bid)
				// + ", addedScore:" + addedScore);
				// }
				for (Issue lIssue : Issues) {
					int issueNum = lIssue.getNumber();
					switch (lIssue.getType()) {
					case DISCRETE:
						IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
						ValueDiscrete value = (ValueDiscrete) bid
								.getValue(lIssue.getNumber());
						int index = lIssueDiscrete.getValueIndex(value);
						issueScore.get(issueNum - 1).set(index,
								issueScore.get(issueNum - 1).get(index)
										+ addedScore);
						break;
					case REAL:
						IssueReal lIssueReal = (IssueReal) lIssue;
						ValueReal valueReal = (ValueReal) bid
								.getValue(lIssue.getNumber());
						int indexReal = (int) ((valueReal.getValue()
								- lIssueReal.getLowerBound())
								/ (lIssueReal.getUpperBound()
										- lIssueReal.getLowerBound())
								* lIssueReal.getNumberOfDiscretizationSteps());
						issueScore.get(issueNum - 1).set(indexReal,
								issueScore.get(issueNum - 1).get(indexReal)
										+ addedScore);
						break;
					case INTEGER:
						IssueInteger lIssueInteger = (IssueInteger) lIssue;
						ValueInteger valueInteger = (ValueInteger) bid
								.getValue(lIssue.getNumber());
						int indexInteger = valueInteger.getValue();
						issueScore.get(issueNum - 1).set(indexInteger,
								issueScore.get(issueNum - 1).get(indexInteger)
										+ addedScore);
						break;
					default:
						throw new Exception("issue type " + lIssue.getType()
								+ " not supported");
					}
				}
			} catch (Exception e) {
				System.out.println(e.getMessage()
						+ "exception in method changeIssueScore");
			}
		}

		private double calculateScoreValue(Bid Bid1) {
			double score = 0;
			try {
				for (Issue Issue1 : Issues) {
					int issueNum = Issue1.getNumber();
					switch (Issue1.getType()) {
					case DISCRETE:
						IssueDiscrete lIssueDiscrete = (IssueDiscrete) Issue1;
						ValueDiscrete value1 = (ValueDiscrete) Bid1
								.getValue(issueNum);
						int index = lIssueDiscrete.getValueIndex(value1);
						if (domainSize > largeDomainSize) {
							score += Math.log(issueScore
									.get(lIssueDiscrete.getNumber() - 1)
									.get(index));
						} else {
							score += issueScore
									.get(lIssueDiscrete.getNumber() - 1)
									.get(index);
						}
						break;

					case REAL:
						IssueReal lIssueReal = (IssueReal) Issue1;
						ValueReal valueReal = (ValueReal) Bid1
								.getValue(issueNum);
						int indexReal = (int) ((valueReal.getValue()
								- lIssueReal.getLowerBound())
								/ (lIssueReal.getUpperBound()
										- lIssueReal.getLowerBound())
								* lIssueReal.getNumberOfDiscretizationSteps());
						if (domainSize > largeDomainSize) {
							score += Math.log(
									issueScore.get(lIssueReal.getNumber() - 1)
											.get(indexReal));
						} else {
							score += issueScore.get(lIssueReal.getNumber() - 1)
									.get(indexReal);
						}
						break;
					case INTEGER:
						IssueInteger lIssueInteger = (IssueInteger) Issue1;
						ValueInteger valueInteger = (ValueInteger) Bid1
								.getValue(issueNum);
						int indexInteger = valueInteger.getValue();
						if (domainSize > largeDomainSize) {
							score += Math.log(issueScore
									.get(lIssueInteger.getNumber() - 1)
									.get(indexInteger));
						} else {
							score += issueScore
									.get(lIssueInteger.getNumber() - 1)
									.get(indexInteger);
						}
						break;
					}
				}
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Bid:" +
				// this.utilitySpace.getUtility(Bid1) +
				// ", NotNormalized IssueScore:" + score);
				// }
			} catch (Exception e) {
				System.out.println(e.getMessage()
						+ " Exception in method calculateScoreValue");
				return 0;
			}
			return score;
		}

		private void calculateBidScore(Bid Bid1) {
			try {
				avgScore = calculateScoreValue(Bid1);
				issueScoreBids.add(avgScore);
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Bid:" + utilitySpace.getUtility(Bid1) +
				// ", AvgScore:" + this.avgScore);
				// }
				if (Math.abs(avgScore) > maxScore) {
					maxScore = Math.abs(avgScore);
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Bid:" +
					// this.utilitySpace.getUtility(Bid1) + ", Max IssueScore:"
					// + this.maxScore);
					// }
				}
			} catch (Exception e) {
				System.out.println(e.getMessage()
						+ " Exception in method calculateBidScore");
			}
		}
	}

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		try {
			// if(debug)
			// {
			// fileName = "OurBiddingData"+hashCode()+".txt";
			// file = new PrintStream(new FileOutputStream(fileName, false));
			// //file.println("\n");
			// }
			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			double count = 0;
			for (; myBidIterator.hasNext();) {
				Bid bid = myBidIterator.next();
				count = count + 1;
			}
			this.domainSize = count;
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("Size:" + this.domainSize);
			// }
			this.largeDomainSize = 1000;
			this.isInProfile3 = false;
			this.isNeededToMoveProfile3 = false;
			this.round = 0;
			this.string_agentData1Num = "1";
			this.string_agentData2Num = "2";
			this.string_AddIssueScore = "addIssueScore";
			this.string_SubtractIssueScore = "subtractIssueScore";
			this.initialConcedeTime = 0.85;
			this.agentData1 = new AgentData(this.reservationValue,
					this.initialConcedeTime, this.initialUtilitythreshold,
					this.string_agentData1Num);
			this.agentData2 = new AgentData(this.reservationValue,
					this.initialConcedeTime, this.initialUtilitythreshold,
					this.string_agentData2Num);
			this.countOpp = 0;
			this.opponentBidHistory1 = new OpponentBidHistory();
			this.opponentBidHistory2 = new OpponentBidHistory();
			this.Issues = this.utilitySpace.getDomain().getIssues();
			this.bid_maximum_utility = this.utilitySpace.getMaxUtilityBid();
			this.bid_minimum_utility = this.utilitySpace.getMinUtilityBid();
			this.MaximumUtility = this.utilitySpace
					.getUtility(this.bid_maximum_utility);
			this.offeredBidsAvg = this.MaximumUtility;
			this.offeredBidsSD = (this.MaximumUtility
					- this.utilitySpace.getUtility(this.bid_minimum_utility))
					/ 4;
			this.offeredBidsRound = 0;
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("OfferedBidsRound:" + this.offeredBidsRound +
			// ", AdaptiveAvg: " + this.offeredBidsAvg + ", AdaptiveSD:" +
			// this.offeredBidsSD);
			// }
			this.reservationValue = utilitySpace.getReservationValue();
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
			this.NumberofDivisions = 20;
			this.calculateBidsBetweenUtility();
			if (this.discountingFactor <= 0.5D) {
				this.chooseConcedeToDiscountingDegree();
			}
			this.opponentBidHistory1
					.initializeDataStructures(utilitySpace.getDomain());
			this.opponentBidHistory2
					.initializeDataStructures(utilitySpace.getDomain());
			this.timeLeftAfter = timeline.getCurrentTime();
			this.concedeToOpponent = false;
			this.toughAgent = false;
			this.alpha1 = 2;

			if (this.domainSize < 100) {
				this.k = 1;
			} else if (this.domainSize < 1000) {
				this.k = 2;
			} else if (this.domainSize < 10000) {
				this.k = 3;
			} else {
				this.k = 4;
			}
			this.FirstTimeInterval = 20;
			this.SecondTimeInterval = 40;
			this.ThirdTimeInterval = 60;
			this.initialUtilitythreshold = 0.5;
			this.avgUtilitythreshold = 0.5;
			this.avgConcedeTime = 0.5;
			this.EachRoundCount = 0;
			this.myLastBid = this.bid_maximum_utility;
			// this.issueScore = new ArrayList<ArrayList<Double>>();
			// this.issueMaxScore = new ArrayList<Double>();
			this.bidsUtilityScore = UpdateBidsUtilityScore(
					this.bidsUtilityScore);
			for (Issue lIssue : this.Issues) {
				ArrayList<Double> templist1 = new ArrayList<Double>();
				ArrayList<Double> templist2 = new ArrayList<Double>();
				// int issueNum = lIssue.getNumber();
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("IssueNum:" + issueNum +
					// ", NumOfValue:" + lIssueDiscrete.getNumberOfValues());
					// }
					for (int i = 0; i < lIssueDiscrete
							.getNumberOfValues(); i++) {
						templist1.add(i, Double.valueOf(1));
						templist2.add(i, Double.valueOf(1));
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("index:" + i);
						// }
					}
					this.agentData1.issueScore.add(templist1);
					this.agentData2.issueScore.add(templist2);
					// this.agentData1.issueMaxScore.add(Double.valueOf(0));
					// this.agentData2.issueMaxScore.add(Double.valueOf(0));
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("End issueNum " + issueNum);
					// }
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					for (int i = 0; i < lIssueReal
							.getNumberOfDiscretizationSteps(); i++) {
						templist1.add(i, Double.valueOf(1));
						templist2.add(i, Double.valueOf(1));
					}
					this.agentData1.issueScore.add(templist1);
					this.agentData2.issueScore.add(templist2);
					// this.agentData1.issueMaxScore.add(Double.valueOf(0));
					// this.agentData2.issueMaxScore.add(Double.valueOf(0));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					for (int i = 0; i < lIssueInteger.getUpperBound()
							- lIssueInteger.getLowerBound() + 1; i++) {
						templist1.add(i, Double.valueOf(1));
						templist2.add(i, Double.valueOf(1));
					}
					this.agentData1.issueScore.add(templist1);
					this.agentData2.issueScore.add(templist2);
					// this.agentData1.issueMaxScore.add(Double.valueOf(0));
					// this.agentData2.issueMaxScore.add(Double.valueOf(0));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported");
				}
			}
		} catch (Exception e) {
			System.out.println("initialization error" + e.getMessage());
		}
	}

	public String getVersion() {
		return "Ngent";
	}

	public String getName() {
		return "Ngent";
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		Action action = null;
		this.EachRoundCount = this.countOpp;
		this.round = this.round + 1;
		try {
			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			this.timeLeftBefore = timeline.getCurrentTime();
			Bid bid = null;
			// we propose first and propose the bid with maximum utility
			if (this.round == 1) {
				bid = this.bid_maximum_utility;
				UpdateOfferedBidsStat(this.utilitySpace.getUtility(bid));
				action = new Offer(getPartyId(), bid);
			}
			// else if (ActionOfOpponent instanceof Offer)
			else {
				// the opponent propose first and we response secondly
				// update opponent model first
				this.opponentBidHistory1.updateOpponentModel(
						this.agentData1.oppPreviousBid,
						utilitySpace.getDomain(), this.utilitySpace);
				this.opponentBidHistory2.updateOpponentModel(
						this.agentData2.oppPreviousBid,
						utilitySpace.getDomain(), this.utilitySpace);
				if (this.discountingFactor == 1) {
					this.agentData1 = updateConcedeDegree_nonDF(
							this.agentData1);
					this.agentData2 = updateConcedeDegree_nonDF(
							this.agentData2);
				} else if (this.discountingFactor <= 0.5) {
					this.agentData1 = updateConcedeDegree_smallDF(
							this.agentData1);
					this.agentData2 = updateConcedeDegree_smallDF(
							this.agentData2);
				} else {
					updateConcedeDegree_largeDF(this.agentData1);
					updateConcedeDegree_largeDF(this.agentData2);
				}
				Bid oppPreBid;
				if (this.isInProfile3) {
					oppPreBid = this.agentData1.oppPreviousBid;
				} else {
					oppPreBid = this.agentData2.oppPreviousBid;
				}
				// update the estimation
				if (estimateRoundLeft(true) > 10) {
					// still have some rounds left to further negotiate (the
					// major negotiation period)
					bid = BidToOffer_original();
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Proposed Bid!");
					// }
					Boolean IsAccept = AcceptOpponentOffer(oppPreBid, bid);
					IsAccept = OtherAcceptCondition(oppPreBid, IsAccept);

					Boolean IsTerminate = TerminateCurrentNegotiation(bid);
					if (IsAccept && !IsTerminate) {
						action = new Accept(getPartyId(), oppPreBid);
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("Accept the offer");
						// }
					} else if (IsTerminate && !IsAccept) {
						action = new EndNegotiation(getPartyId());
						// System.out.println("we determine to terminate the
						// negotiation");
					} else if (IsAccept && IsTerminate) {
						if (this.utilitySpace.getUtility(
								oppPreBid) > this.reservationValue) {
							action = new Accept(getPartyId(), oppPreBid);
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Compare Accept & Terminate,
							// we accept the offer");
							// }
						} else {
							action = new EndNegotiation(getPartyId());
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Compare Accept & Terminate,
							// we terminate the negotiation");
							// }
						}
					} else {
						// we expect that the negotiation is over once we select
						// a bid from the opponent's history.
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
							} else if (this.utilitySpace.getUtility(
									bid1) == this.utilitySpace.getUtility(bid2)
									&& Math.random() < 0.5) {
								bid = bid2;
							} else {
								bid = bid1;
							}
							action = new Offer(getPartyId(), bid);
							// System.out.println("we offer the best bid in the
							// history and the opponent should accept it");
							this.toughAgent = true;
							this.concedeToOpponent = false;
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("we offer the best bid in the
							// history and the opponent should accept it. Bid:"
							// + this.utilitySpace.getUtility(bid));
							// }
						} else {
							action = new Offer(getPartyId(), bid);
							this.toughAgent = false;
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Propose Bid: " +
							// this.utilitySpace.getUtility(bid) + " at time " +
							// timeline.getTime());
							// }
						}
					}
				} else {// this is the last chance and we concede by providing
						// the opponent the best offer he ever proposed to us
						// in this case, it corresponds to an opponent whose
						// decision time is short

					if (timeline.getTime() > 0.99
							&& estimateRoundLeft(true) < 5) {
						if (this.utilitySpace.getUtility(
								oppPreBid) > this.avgUtilitythreshold) {
							action = new Accept(getPartyId(), oppPreBid);
							bid = oppPreBid;
						} else {
							bid = BidToOffer_original();

							Boolean IsAccept = AcceptOpponentOffer(oppPreBid,
									bid);
							IsAccept = OtherAcceptCondition(oppPreBid,
									IsAccept);

							Boolean IsTerminate = TerminateCurrentNegotiation(
									bid);
							if (IsAccept && !IsTerminate) {
								action = new Accept(getPartyId(), oppPreBid);
								// System.out.println("accept the offer");
							} else if (IsTerminate && !IsAccept) {
								action = new EndNegotiation(getPartyId());
								// System.out.println("we determine to terminate
								// the negotiation");
							} else if (IsTerminate && IsAccept) {
								if (this.utilitySpace.getUtility(
										oppPreBid) > this.reservationValue) {
									action = new Accept(getPartyId(),
											oppPreBid);
									// System.out.println("we accept the offer
									// RANDOMLY");
								} else {
									action = new EndNegotiation(getPartyId());
									// System.out.println("we determine to
									// terminate the negotiation RANDOMLY");
								}
							} else {
								if (this.toughAgent == true) {
									action = new Accept(getPartyId(),
											oppPreBid);
									// System.out.println("the opponent is tough
									// and the deadline is approching thus we
									// accept the offer");
								} else {
									action = new Offer(getPartyId(), bid);
									// this.toughAgent = true;
									// System.out.println("this is really the
									// last chance"
									// + bid.toString() + " with utility of " +
									// utilitySpace.getUtility(bid));
								}
							}
							// in this case, it corresponds to the situation
							// that we encounter an opponent who needs more
							// computation to make decision each round
						}
					} else {// we still have some time to negotiate,
							// and be tough by sticking with the lowest one in
							// previous offer history.
							// we also have to make the decisin fast to avoid
							// reaching the deadline before the decision is made
							// bid = ownBidHistory.GetMinBidInHistory();//reduce
							// the computational cost
						bid = BidToOffer_original();
						// System.out.println("test----------------------------------------------------------"
						// + timeline.getTime());
						Boolean IsAccept = AcceptOpponentOffer(oppPreBid, bid);
						IsAccept = OtherAcceptCondition(oppPreBid, IsAccept);

						Boolean IsTerminate = TerminateCurrentNegotiation(bid);
						if (IsAccept && !IsTerminate) {
							action = new Accept(getPartyId(), oppPreBid);
							// System.out.println("accept the offer");
						} else if (IsTerminate && !IsAccept) {
							action = new EndNegotiation(getPartyId());
							// System.out.println("we determine to terminate the
							// negotiation");
						} else if (IsAccept && IsTerminate) {
							if (this.utilitySpace.getUtility(
									oppPreBid) > this.reservationValue) {
								action = new Accept(getPartyId(), oppPreBid);
								// System.out.println("we accept the offer
								// RANDOMLY");
							} else {
								action = new EndNegotiation(getPartyId());
								// System.out.println("we determine to terminate
								// the negotiation RANDOMLY");
							}
						} else {
							action = new Offer(getPartyId(), bid);
							// System.out.println("we have to be tough now" +
							// bid.toString() + " with utility of " +
							// utilitySpace.getUtility(bid));
						}
					}
				}
			}
			this.myLastBid = bid;

			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			// this.ownBidHistory.addBid(bid, utilitySpace);
			this.timeLeftAfter = timeline.getCurrentTime();
			if (this.timeLeftAfter
					- this.timeLeftBefore > this.maximumTimeOfOwn) {
				this.maximumTimeOfOwn = this.timeLeftAfter
						- this.timeLeftBefore;
			} // update the estimation
				// System.out.println(this.utilitythreshold + "-***-----" +
				// this.timeline.getElapsedSeconds());

			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// //System.out.println("Size: " + this.numberOfBids + ". Res: " +
			// Double.toString(this.reservationValue) + ". Dis: " +
			// Double.toString(this.discountingFactor) + "\n");
			// System.out.println("Round:" + this.round + ", Time: " +
			// this.timeline.getTime()+", Avg:" +
			// this.avgUtilitythreshold+", MinAccept:" +
			// this.minUtilityUhreshold);
			// System.out.println("NormalMinThreAgent1: " +
			// this.agentData1.minThreshold + ", NormalMinThreAgent2: " +
			// this.agentData2.minThreshold);
			// System.out.println("AdaptiveMinU: " + this.adaptiveMinUThreshold
			// + ", finalMinUAgent1:" + this.finalMinUAgent1 +
			// ", finalMinUAgent2:" + this.finalMinUAgent2);
			// }
		} catch (Exception e) {
			System.out.println("Exception in chooseAction:" + e.getMessage());
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

		try {
			this.countOpp = this.countOpp + 1;
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("Enter receiveMessage! round:" + round +
			// ", CountOpp:" + countOpp + ", EachRoundCount:" + EachRoundCount);
			// }
			if (this.countOpp == this.EachRoundCount + 3) {
				this.EachRoundCount = this.EachRoundCount + 2;
				this.isNeededToMoveProfile3 = true;
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Abnormal! round:" + round + ", CountOpp:"
				// + countOpp + ", EachRoundCount:" + EachRoundCount);
				// }
			}
			if (!this.isInProfile3) {
				// In Profile 1 or 2
				System.out.println("Current Profile: 1 or 2");
				if (this.countOpp == this.EachRoundCount + 1)// Next Agent
																// w.r.t. us
				{
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Enter AgentData1! round:" + round +
					// ", CountOpp:" + countOpp + ", EachRoundCount:" +
					// EachRoundCount);
					// }
					if ((this.ActionOfOpponent instanceof Offer)) {
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("AgentData1 Action: Offer! round:"
						// + round + ", CountOpp:" + countOpp +
						// ", EachRoundCount:" + EachRoundCount);
						// }
						if (!this.agentData1.GetIsReceivedOppFirstBid()) {
							this.agentData1.oppFirstBid = ((Offer) this.ActionOfOpponent)
									.getBid();
							this.agentData1.oppFirstBidUtility = this.utilitySpace
									.getUtility(this.agentData1.oppFirstBid);
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Enter SetFirstBidInform
							// AgentData1! round:"
							// + round + ", CountOpp:" + countOpp +
							// ", EachRoundCount:" + EachRoundCount);
							// }
							this.agentData1.SetIsReceivedOppFirstBid(true);
						}
						this.agentData1.oppPreviousBid = ((Offer) this.ActionOfOpponent)
								.getBid();
						this.agentData1.ChangeIssueScore(
								this.agentData1.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData1.oppPreviousBid));
					} else if ((this.ActionOfOpponent instanceof Accept)) {
						this.agentData1.oppPreviousBid = this.myLastBid;
						this.agentData1.ChangeIssueScore(
								this.agentData1.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData1.oppPreviousBid));
					}
				} else if (this.countOpp == this.EachRoundCount + 2) // Previous
																		// Agent
																		// w.r.t.
																		// us
				{
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Enter AgentData2! round:" + round +
					// ", CountOpp:" + countOpp + ", EachRoundCount:" +
					// EachRoundCount);
					// }
					if ((this.ActionOfOpponent instanceof Offer)) {
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("AgentData2 Action: Offer! round:"
						// + round + ", CountOpp:" + countOpp +
						// ", EachRoundCount:" + EachRoundCount);
						// }
						if (!this.agentData2.GetIsReceivedOppFirstBid()) {
							this.agentData2.oppFirstBid = ((Offer) this.ActionOfOpponent)
									.getBid();
							this.agentData2.oppFirstBidUtility = this.utilitySpace
									.getUtility(this.agentData2.oppFirstBid);
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Enter SetFirstBidInform
							// AgentData2! round:"
							// + round + ", CountOpp:" + countOpp +
							// ", EachRoundCount:" + EachRoundCount);
							// }
							this.agentData2.SetIsReceivedOppFirstBid(true);
						}
						this.agentData2.oppPreviousBid = ((Offer) this.ActionOfOpponent)
								.getBid();
						this.agentData2.ChangeIssueScore(
								this.agentData2.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData2.oppPreviousBid));
					} else if ((this.ActionOfOpponent instanceof Accept)) {
						this.agentData2.oppPreviousBid = this.agentData1.oppPreviousBid;
						this.agentData2.ChangeIssueScore(
								this.agentData2.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData2.oppPreviousBid));
					}
				}

				if (this.isNeededToMoveProfile3) {
					this.isInProfile3 = true;
				}
			} else {
				// In Profile 3
				System.out.println("Current Profile: 3");
				if (this.countOpp == this.EachRoundCount + 2)// Previous Agent
																// w.r.t. us
				{
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Enter AgentData1! round:" + round +
					// ", CountOpp:" + countOpp + ", EachRoundCount:" +
					// EachRoundCount);
					// }
					if ((this.ActionOfOpponent instanceof Offer)) {
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("AgentData1 Action: Offer! round:"
						// + round + ", CountOpp:" + countOpp +
						// ", EachRoundCount:" + EachRoundCount);
						// }
						if (!this.agentData1.GetIsReceivedOppFirstBid()) {
							this.agentData1.oppFirstBid = ((Offer) this.ActionOfOpponent)
									.getBid();
							this.agentData1.oppFirstBidUtility = this.utilitySpace
									.getUtility(this.agentData1.oppFirstBid);
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Enter SetFirstBidInform
							// AgentData1! round:"
							// + round + ", CountOpp:" + countOpp +
							// ", EachRoundCount:" + EachRoundCount);
							// }
							this.agentData1.SetIsReceivedOppFirstBid(true);
						}
						this.agentData1.oppPreviousBid = ((Offer) this.ActionOfOpponent)
								.getBid();
						this.agentData1.ChangeIssueScore(
								this.agentData1.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData1.oppPreviousBid));
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("AgentData1 Action: Offer! Bid:" +
						// this.utilitySpace.getUtility(this.agentData1.oppPreviousBid));
						// }
					} else if ((this.ActionOfOpponent instanceof Accept)) {
						this.agentData1.oppPreviousBid = this.agentData2.oppPreviousBid;
						this.agentData1.ChangeIssueScore(
								this.agentData1.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData1.oppPreviousBid));
					}
				} else if (this.countOpp == this.EachRoundCount + 1) // Next
																		// Agent
																		// w.r.t.
																		// us
				{
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Enter AgentData2! round:" + round +
					// ", CountOpp:" + countOpp + ", EachRoundCount:" +
					// EachRoundCount);
					// }
					if ((this.ActionOfOpponent instanceof Offer)) {
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("AgentData2 Action: Offer! round:"
						// + round + ", CountOpp:" + countOpp +
						// ", EachRoundCount:" + EachRoundCount);
						// }
						if (!this.agentData2.GetIsReceivedOppFirstBid()) {
							this.agentData2.oppFirstBid = ((Offer) this.ActionOfOpponent)
									.getBid();
							this.agentData2.oppFirstBidUtility = this.utilitySpace
									.getUtility(this.agentData2.oppFirstBid);
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Enter SetFirstBidInform
							// AgentData2! round:"
							// + round + ", CountOpp:" + countOpp +
							// ", EachRoundCount:" + EachRoundCount);
							// }
							this.agentData2.SetIsReceivedOppFirstBid(true);
						}
						this.agentData2.oppPreviousBid = ((Offer) this.ActionOfOpponent)
								.getBid();
						this.agentData2.ChangeIssueScore(
								this.agentData2.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData2.oppPreviousBid));
					} else if ((this.ActionOfOpponent instanceof Accept)) {
						this.agentData2.oppPreviousBid = this.myLastBid;
						this.agentData2.ChangeIssueScore(
								this.agentData2.oppPreviousBid,
								this.string_AddIssueScore);
						UpdateOfferedBidsStat(this.utilitySpace
								.getUtility(this.agentData2.oppPreviousBid));
					}
				}
			}
		} catch (Exception e) {
			System.out.println(
					e.getMessage() + " exception in method receiveMessage");
		}
	}

	private void UpdateOfferedBidsStat(double bidU) {
		try {
			int newOfferedBidsRound = this.offeredBidsRound + 1;
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("OfferedBidsRound:" + this.offeredBidsRound +
			// ", AdaptiveAvg: " + this.offeredBidsAvg + ", AdaptiveSD:" +
			// this.offeredBidsSD + ", bidU:" + bidU);
			// }
			this.offeredBidsAvg = (bidU - this.offeredBidsAvg)
					/ newOfferedBidsRound + this.offeredBidsAvg;
			this.offeredBidsSD = Math.sqrt((this.offeredBidsRound
					* this.offeredBidsSD * this.offeredBidsSD
					+ (bidU - this.offeredBidsAvg)
							* (bidU - this.offeredBidsAvg))
					/ newOfferedBidsRound);
			this.offeredBidsRound = newOfferedBidsRound;
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("OfferedBidsRound:" + this.offeredBidsRound +
			// ", AdaptiveAvg: " + this.offeredBidsAvg + ", AdaptiveSD:" +
			// this.offeredBidsSD + ", bidU:" + bidU);
			// }
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method UpdateOfferedBidsStat");
		}
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
			if (currentUtility >= this.avgUtilitythreshold
					|| currentUtility >= nextRoundUtility) {
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("currentUtility >=
				// this.avgUtilitythreshold || currentUtility >=
				// nextRoundUtility: true, AvgU:"
				// + this.avgUtilitythreshold + ", currentU:" + currentUtility +
				// ", nextRoundUtility:" + nextRoundUtility);
				// }
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
						&& timeline.getTime() > this.avgConcedeTime) {
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("currentMaximumUtility >
					// predictMaximumUtility && timeline.getTime() >
					// this.avgConcedeTime: true, currentMaximumUtility:"
					// + currentMaximumUtility + ", predictMaximumUtility:" +
					// predictMaximumUtility + ", avgConcedeTime:" +
					// this.avgConcedeTime);
					// }
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace.getUtilityWithDiscount(opponentBid,
							timeline) >= currentMaximumUtility - 0.01) {
						// System.out.println("he offered me " +
						// currentMaximumUtility +
						// " we predict we can get at most " +
						// predictMaximumUtility +
						// "we concede now to avoid lower payoff due to
						// conflict");
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("utilitySpace.getUtilityWithDiscount(opponentBid,
						// timeline) >= currentMaximumUtility - 0.01: true,
						// DFoppBid:"
						// + utilitySpace.getUtilityWithDiscount(opponentBid,
						// timeline) + ", currentMaximumUtility - 0.01:" +
						// (currentMaximumUtility - 0.01));
						// }
						return true;
					} else {
						this.concedeToOpponent = true;
						return false;
					}
				}
				// retrieve the opponent's biding history and utilize it
				else if (currentMaximumUtility > this.avgUtilitythreshold * Math
						.pow(this.discountingFactor, timeline.getTime())) {
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("currentMaximumUtility >
					// this.avgUtilitythreshold *
					// Math.pow(this.discountingFactor, timeline.getTime()):
					// true, currentMaximumUtility:"
					// + currentMaximumUtility + ", avgU:" + avgUtilitythreshold
					// + ", DFavgU:" + this.avgUtilitythreshold *
					// Math.pow(this.discountingFactor, timeline.getTime()));
					// }
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace.getUtilityWithDiscount(opponentBid,
							timeline) >= currentMaximumUtility - 0.01) {
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("utilitySpace.getUtilityWithDiscount(opponentBid,
						// timeline) >= currentMaximumUtility - 0.01: true,
						// DFoppBid:"
						// + utilitySpace.getUtilityWithDiscount(opponentBid,
						// timeline) + ", currentMaximumUtility - 0.01:" +
						// (currentMaximumUtility - 0.01));
						// }
						return true;
					} else {
						// System.out.println("test" +
						// utilitySpace.getUtility(opponentBid) +
						// this.AvgUtilitythreshold);
						this.concedeToOpponent = true;
						return false;
					}
				} else {
					return false;
				}
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method AcceptOpponentOffer");
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
						&& timeline.getTime() > this.avgConcedeTime) {
					return true;
				} else {
					return false;
				}
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method TerminateCurrentNegotiation");
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
			}
			round = (this.totalTime - timeline.getCurrentTime())
					/ (this.maximumTimeOfOpponent + this.maximumTimeOfOwn);
			// System.out.println("current time is " +
			// timeline.getElapsedSeconds() + "---" + round + "----" +
			// this.maximumTimeOfOpponent);
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method TerminateCurrentNegotiation");
			return 20;
		}
		return ((int) (round));
	}

	/*
	 * pre-processing to save the computational time each round
	 */
	private void calculateBidsBetweenUtility() {
		try {
			this.MinimumUtility = Math.min(this.agentData1.MinimumUtility,
					this.agentData2.MinimumUtility);
			this.SpacebetweenDivisions = (this.MaximumUtility
					- this.MinimumUtility) / this.NumberofDivisions;
			// initalization for each LinkedList storing the bids between each
			// range
			this.bidsBetweenUtility = new LinkedList<LinkedList<Bid>>();
			for (int i = 0; i < this.NumberofDivisions; i++) {
				LinkedList<Bid> BidList = new LinkedList<Bid>();
				// BidList.add(this.bid_maximum_utility);
				this.bidsBetweenUtility.add(i, BidList);
			}
			// this.bidsBetweenUtility.get(this.NumberofDivisions-1).add(this.bid_maximum_utility);
			// note that here we may need to use some trick to reduce the
			// computation cost (to be checked later);
			// add those bids in each range into the corresponding LinkedList
			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			while (myBidIterator.hasNext()) {
				Bid b = myBidIterator.next();
				for (int i = 0; i < this.NumberofDivisions; i++) {
					if (this.utilitySpace.getUtility(
							b) <= (i + 1) * this.SpacebetweenDivisions
									+ this.MinimumUtility
							&& this.utilitySpace.getUtility(
									b) >= i * this.SpacebetweenDivisions
											+ this.MinimumUtility) {
						this.bidsBetweenUtility.get(i).add(b);
						break;
					}
				}
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method calculateBidsBetweenUtility");
		}
	}

	private LinkedList<Bid> calculateBidsAboveUtility(double bidutil) {
		BidIterator myBidIterator = null;
		myBidIterator = new BidIterator(this.utilitySpace.getDomain());

		// initalization for each LinkedList storing the bids between each range
		LinkedList<Bid> Bids = new LinkedList<Bid>();
		try {
			// note that here we may need to use some trick to reduce the
			// computation cost (to be checked later);
			// add those bids in each range into the corresponding LinkedList
			while (myBidIterator.hasNext()) {
				Bid b = myBidIterator.next();
				if (this.utilitySpace.getUtility(b) > bidutil) {
					Bids.add(b);
				}
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method calculateBidsAboveUtility");
		}
		return Bids;
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
				throw new Exception("issue type " + lIssue.getType()
						+ " not supported. Exception in method RandomSearchBid");
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
		List<Bid> bidsInRange = new LinkedList<Bid>();
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
			System.out.println(e.getMessage()
					+ " Exception in method getBidsBetweenUtility");
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
			this.avgConcedeTime = this.minConcedeTime
					+ (1 - this.minConcedeTime) * alpha;
			this.concedeTime_original = this.avgConcedeTime;
			// System.out.println("concedeToDiscountingFactor is " +
			// this.AvgConcedeTime + "current time is " + timeline.getTime());
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method chooseConcedeToDiscountingDegree");
		}
	}

	/*
	 * update the concede-to-time degree based on the predicted toughness degree
	 * of the opponent
	 */

	private Bid regeneratebid(double bidUtil) {
		Bid ans = null;
		LinkedList<Bid> Bids = new LinkedList<Bid>();
		// LinkedList <Bid> Bids = calculateBidsAboveUtility(bidutil);
		this.agentData1.maxScore = -1;
		this.agentData2.maxScore = -1;
		this.agentData1.avgScore = 0;
		this.agentData2.avgScore = 0;
		try {
			// if(this.OppFinalBids1.get(this.storingcapacity - 1) == null ||
			// this.OppFinalBids2.get(this.storingcapacity - 1) == null)
			// {
			// ans = this.bid_maximum_utility;
			// }
			// else
			// {
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("----regeneatebid---- bidUtil:" + bidUtil);
			// }
			// if(this.numberOfBids > 1000)
			// {
			int check = 0;
			for (this.CurrentrangeNumber = this.NumberofDivisions
					- 1; check == 0; this.CurrentrangeNumber--) {
				if (bidUtil > ((this.CurrentrangeNumber + 1)
						* this.SpacebetweenDivisions + this.MinimumUtility)) {
					break;
				}
				Bids.addAll(
						this.bidsBetweenUtility.get(this.CurrentrangeNumber));
			}
			// }
			// else
			// {
			// BidIterator myBidIterator = new
			// BidIterator(this.utilitySpace.getDomain());
			// for (;myBidIterator.hasNext();)
			// {
			// Bid bid = myBidIterator.next();
			// if(bidUtil <= this.utilitySpace.getUtility(bid))
			// {
			// // if(debug)
			// // {
			// // file = new PrintStream(new FileOutputStream(fileName, true));
			// // System.setOut(file);
			// // System.out.println("myBidIterator:" +
			// this.utilitySpace.getUtility(bid));
			// // }
			// Bids.add(bid);
			// }
			// }
			// }
			if (Bids.isEmpty()) {
				ans = this.myLastBid;
			} else {
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Enter loop!");
				// }
				LinkedList<Bid> Bidsconsider = new LinkedList<Bid>();
				if (Bids.size() >= this.maxNumOfBids) {
					int i = 0;
					while (i < this.maxNumOfBids) {
						// boolean isContinue = true;
						// int oldSize = Bids.size();
						int ran = (int) (Math.random() * Bids.size());
						// for(Bid rejectedBid: this.RejectedBids)
						// {
						// //if(this.utilitySpace.getUtility(Bids.get(ran)) ==
						// this.utilitySpace.getUtility(rejectedBid))
						// if(Bids.get(ran).equals(rejectedBid))
						// {
						// Bids.remove(ran);
						// isContinue = false;
						// break;
						// }
						// }
						// Bids = removeRejectedBid(Bids, ran);
						// if(isContinue)
						// {
						Bidsconsider.add(Bids.remove(ran));
						// }
						i++;
					}
					Bids = Bidsconsider;
				}
				// else
				// {
				// for(int i = 0; i < Bids.size(); i++)
				// {
				// boolean isContinue = true;
				// for(Bid rejectedBid: this.RejectedBids)
				// {
				// //if(this.utilitySpace.getUtility(Bids.get(ran)) ==
				// this.utilitySpace.getUtility(rejectedBid))
				// if(Bids.get(i).equals(rejectedBid))
				// {
				// isContinue = false;
				// break;
				// }
				// }
				// if(isContinue)
				// {
				// Bidsconsider.add(Bids.get(i));
				// }
				// }
				// }
				// Bids = Bidsconsider;
				// if(debug)
				// {
				// System.out.println(">0.95!");
				// }
				if (Bids.isEmpty()) {
					ans = this.myLastBid;
				} else {
					this.agentData1.issueScoreBids = new LinkedList<Double>();
					this.agentData2.issueScoreBids = new LinkedList<Double>();
					// this.agentData1.ChangeIssueMaxScore();
					// this.agentData2.ChangeIssueMaxScore();
					for (Bid Bid1 : Bids) {
						// if(debug)
						// {
						// System.out.println("Enter second for loop! i=" + i);
						// }
						// Cal Total score From all Bids and add it to
						// agentData's List
						this.agentData1.calculateBidScore(Bid1);
						this.agentData2.calculateBidScore(Bid1);
					}
					double min = 1.1;
					double dist;
					LinkedList<Double> distList = new LinkedList<Double>();
					for (int i = 0; i < Bids.size(); i++) {
						// temp = (Issuescorebid.get(i)/this.maxScore * 1 / 5) +
						// (this.bidsUtilityScore.get(Bids.get(i)) * 4 / 5);
						Bid bid = Bids.get(i);
						double agent1BidScore = this.agentData1.issueScoreBids
								.get(i) / this.agentData1.maxScore;
						double agent2BidScore = this.agentData2.issueScoreBids
								.get(i) / this.agentData2.maxScore;
						double ourBidScore = this.utilitySpace.getUtility(bid);
						dist = (1 - agent1BidScore) * (1 - agent1BidScore)
								+ (1 - agent2BidScore) * (1 - agent2BidScore)
								+ (1 - ourBidScore) * (1 - ourBidScore);
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println("Bid:" +
						// this.utilitySpace.getUtility(bid) + ", dist:" +
						// dist);
						// System.out.println("agent1BidScore:" +
						// this.agentData1.issueScoreBids.get(i)
						// +"/"+this.agentData1.maxScore + " = " +
						// agent1BidScore);
						// System.out.println("agent2BidScore:" +
						// this.agentData2.issueScoreBids.get(i)
						// +"/"+this.agentData2.maxScore + " = " +
						// agent2BidScore);
						// }
						distList.add(dist);
						if (dist < min) {
							min = dist;
							ans = bid;
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("Ans Bid:" +
							// this.utilitySpace.getUtility(ans) +
							// ", Max Score:" + max);
							// for (Issue Issue1 : this.Issues)
							// {
							// int issueNum = Issue1.getNumber();
							// switch (Issue1.getType())
							// {
							// case DISCRETE:
							// IssueDiscrete lIssueDiscrete =
							// (IssueDiscrete)Issue1;
							// ValueDiscrete value1 = (ValueDiscrete)
							// ans.getValue(issueNum);
							// int index = lIssueDiscrete.getValueIndex(value1);
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("DISCRETE - IssueNum:" +
							// issueNum + ", Index:" + index + ", Score:" +
							// this.issueScore.get(lIssueDiscrete.getNumber() -
							// 1).get(index) /
							// this.issueMaxScore.get(lIssueDiscrete.getNumber()
							// - 1));
							// }
							// break;
							// case REAL:
							// IssueReal lIssueReal = (IssueReal) Issue1;
							// ValueReal valueReal = (ValueReal)
							// ans.getValue(issueNum);
							// int indexReal = (int)((valueReal.getValue() -
							// lIssueReal.getLowerBound()) /
							// (lIssueReal.getUpperBound() -
							// lIssueReal.getLowerBound()) *
							// lIssueReal.getNumberOfDiscretizationSteps());
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("REAL - IssueNum:" + issueNum
							// + ", Index:" + indexReal + ", Score:" +
							// this.issueScore.get(lIssueReal.getNumber() -
							// 1).get(indexReal) /
							// this.issueMaxScore.get(lIssueReal.getNumber() -
							// 1));
							// }
							// break;
							// case INTEGER:
							// IssueInteger lIssueInteger = (IssueInteger)
							// Issue1;
							// ValueInteger valueInteger = (ValueInteger)
							// ans.getValue(issueNum);
							// int indexInteger = valueInteger.getValue();
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("INTEGER - IssueNum:" +
							// issueNum + ", Index:" + indexInteger + ", Score:"
							// + this.issueScore.get(lIssueInteger.getNumber() -
							// 1).get(indexInteger).doubleValue() /
							// this.issueMaxScore.get(lIssueInteger.getNumber()
							// - 1));
							// }
							// break;
							// }
							// }
							// }

						} else if (dist == min && this.utilitySpace.getUtility(
								bid) > this.utilitySpace.getUtility(ans)) {
							ans = bid;
						}
					}

					if (Math.random() > 0.95) {
						LinkedList<Bid> sortedBidsList = SortBidsList(Bids,
								distList);
						ans = sortedBidsList.get(sortedBidsList.size() / 2
								+ (int) (Math.random() * (Bids.size()
										- sortedBidsList.size() / 2)));
						// if(debug)
						// {
						// file = new PrintStream(new FileOutputStream(fileName,
						// true));
						// System.setOut(file);
						// System.out.println(">0.95! rand ans:" +
						// this.utilitySpace.getUtility(ans));
						// }
					}
				}
			}
			// }
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("ans:" + this.utilitySpace.getUtility(ans));
			// }
		} catch (Exception e) {
			System.out.println(
					e.getMessage() + " Exception in method regeneratebid");
			ans = this.bid_maximum_utility;
		}

		return ans;
	}

	private HashMap<Bid, Double> UpdateBidsUtilityScore(
			HashMap<Bid, Double> bidsUtilityScore) {
		try {
			bidsUtilityScore = new HashMap<Bid, Double>();

			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			for (; myBidIterator.hasNext();) {
				Bid bid = myBidIterator.next();
				bidsUtilityScore.put(bid, (this.utilitySpace.getUtility(bid)
						/ this.MaximumUtility));
			}
		} catch (Exception e) {
			System.out.println(
					"Exception in UpdateBidsUtilityScore: " + e.getMessage());
		}
		return bidsUtilityScore;
	}

	private AgentData updateConcedeDegree_smallDF(AgentData agentData) {
		double gama = 10;
		double weight = 0.1;
		double opponnetToughnessDegree = 1;

		try {
			if (agentData.agentNum.equals(this.string_agentData1Num)) {
				opponnetToughnessDegree = this.opponentBidHistory1
						.getConcessionDegree();
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("agentData1 - getConcessionDegree:" +
				// opponnetToughnessDegree);
				// }
			} else {
				opponnetToughnessDegree = this.opponentBidHistory2
						.getConcessionDegree();
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("agentData2 - getConcessionDegree:" +
				// opponnetToughnessDegree);
				// }
			}

			agentData = updateConcedeDegree_UpdateMinU(agentData, "Small");
			double temp = this.concedeTime_original
					+ weight * (1 - this.concedeTime_original)
							* Math.pow(opponnetToughnessDegree, gama);
			agentData = updateConcedeDegree_smallDF_UpdateConcedeTime(agentData,
					temp);
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("concedeTime:" + agentData.concedeTime);
			// }
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " Exception in method updateConcedeDegree_smallDF");
		}
		// System.out.println("concedeToDiscountingFactor is " +
		// this.concedeToDiscountingFactor + "current time is " +
		// timeline.getTime() + "original concedetodiscoutingfactor is " +
		// this.concedeToDiscountingFactor_original);
		return agentData;
	}

	private AgentData updateConcedeDegree_smallDF_UpdateConcedeTime(
			AgentData agentData, double newConcedeTime) {
		try {
			agentData.concedeTime = newConcedeTime;
			if (agentData.concedeTime >= 1) {
				agentData.concedeTime = 1;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method updateConcedeDegree_smallDF_UpdateConcedeTime");
		}
		// System.out.println("concedeToDiscountingFactor is " +
		// this.concedeToDiscountingFactor + "current time is " +
		// timeline.getTime() + "original concedetodiscoutingfactor is " +
		// this.concedeToDiscountingFactor_original);
		return agentData;
	}

	private AgentData updateConcedeDegree_UpdateMinU(AgentData agentData,
			String DFname) {
		try {
			if (agentData.IsOppFirstBid) {
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Enter updateConcedeDegree_UpdateMinU");
				// }
				if (DFname.equals("Small")) {
					if (this.reservationValue <= agentData.oppFirstBidUtility) {
						agentData.MinimumUtility = agentData.oppFirstBidUtility;
					}
				} else {
					LinkedList<Double> bidsUtil = new LinkedList();
					LinkedList<Bid> Bids = new LinkedList<Bid>();
					BidIterator myBidIterator = new BidIterator(
							this.utilitySpace.getDomain());
					for (; myBidIterator.hasNext();) {
						Bid bid = myBidIterator.next();
						bidsUtil.add(this.utilitySpace.getUtility(bid));
						Bids.add(bid);
					}
					double relSumUtility = 0;
					double relCountUtility = 0;

					if (this.domainSize > 100) {
						for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
							double bidUtil = (Double) e.next();
							if (agentData.utility_FirstMaximum < bidUtil) {
								agentData.utility_FirstMaximum = bidUtil;
							} else if (agentData.utility_SecondMaximum < bidUtil) {
								agentData.utility_SecondMaximum = bidUtil;
							}

							if (bidUtil >= agentData.oppFirstBidUtility) {
								relSumUtility += bidUtil;
								relCountUtility += 1;
							}
						}
						agentData = calMinU(bidsUtil, agentData, relSumUtility,
								relCountUtility);
					} else {
						double relAvgUtility = 0;
						double relSqRootOfAvgUtility = 0;

						for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
							double bidUtil = (Double) e.next();
							if (agentData.utility_FirstMaximum < bidUtil) {
								agentData.utility_FirstMaximum = bidUtil;
							} else if (agentData.utility_SecondMaximum < bidUtil) {
								agentData.utility_SecondMaximum = bidUtil;
							}

							if (bidUtil >= agentData.oppFirstBidUtility) {
								relSumUtility += bidUtil;
								relCountUtility += 1;
							}
						}

						if (relCountUtility > (bidsUtil.size() / 2)) {
							agentData = calMinU(bidsUtil, agentData,
									relSumUtility, relCountUtility);
						} else {
							// if(debug)
							// {
							// file = new PrintStream(new
							// FileOutputStream(fileName, true));
							// System.setOut(file);
							// System.out.println("updateConcedeDegree_UpdateMinU:
							// numOfBids < 100, above medium:");
							// }

							LinkedList<Bid> sortedBidsList = SortBidsList(Bids);
							for (int i = sortedBidsList.size()
									/ 2; i < sortedBidsList.size(); i++) {
								double bidUtil = this.utilitySpace
										.getUtility(sortedBidsList.get(i));
								// if(debug)
								// {
								// file = new PrintStream(new
								// FileOutputStream(fileName, true));
								// System.setOut(file);
								// System.out.println("Bid Utility:" + bidUtil);
								// }
								if (agentData.utility_FirstMaximum < bidUtil) {
									agentData.utility_FirstMaximum = bidUtil;
								} else if (agentData.utility_SecondMaximum < bidUtil) {
									agentData.utility_SecondMaximum = bidUtil;
								}

								relSumUtility += bidUtil;
								relCountUtility += 1;
							}

							relAvgUtility = relSumUtility / relCountUtility;
							for (Iterator f = bidsUtil.iterator(); f
									.hasNext();) {
								double bidUtil = (Double) f.next();

								if (bidUtil >= agentData.oppFirstBidUtility) {
									relSqRootOfAvgUtility += (bidUtil
											- relAvgUtility)
											* (bidUtil - relAvgUtility);
								}
							}
							agentData.relStdevUtility = Math.sqrt(
									relSqRootOfAvgUtility / relCountUtility);
							agentData.minThreshold = relAvgUtility;
						}
					}

					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Before minThreshold:" +
					// agentData.minThreshold);
					// }
					agentData.minThreshold = CalWhetherOppFirstBidGoodEnough(
							agentData.minThreshold,
							agentData.oppFirstBidUtility);
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("Middle minThreshold:" +
					// agentData.minThreshold);
					// }
					agentData.minThreshold = Compare_MinThreshold_And_SecondMax(
							agentData.minThreshold,
							agentData.utility_SecondMaximum);
					// if(debug)
					// {
					// file = new PrintStream(new FileOutputStream(fileName,
					// true));
					// System.setOut(file);
					// System.out.println("After minThreshold:" +
					// agentData.minThreshold);
					// }
				}
				agentData.IsOppFirstBid = false;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method updateConcedeDegree_UpdateMinU");
		}
		// System.out.println("concedeToDiscountingFactor is " +
		// this.concedeToDiscountingFactor + "current time is " +
		// timeline.getTime() + "original concedetodiscoutingfactor is " +
		// this.concedeToDiscountingFactor_original);
		return agentData;
	}

	private AgentData calMinU(LinkedList<Double> bidsUtil, AgentData agentData,
			double relSumUtility, double relCountUtility) {
		try {
			double relAvgUtility = 0;
			double relSqRootOfAvgUtility = 0;
			double relStdevUtility = 0;

			relAvgUtility = relSumUtility / relCountUtility;

			for (Iterator f = bidsUtil.iterator(); f.hasNext();) {
				double bidUtil = (Double) f.next();

				if (bidUtil >= agentData.oppFirstBidUtility) {
					relSqRootOfAvgUtility += (bidUtil - relAvgUtility)
							* (bidUtil - relAvgUtility);
				}
			}
			relStdevUtility = Math
					.sqrt(relSqRootOfAvgUtility / relCountUtility);

			if (relStdevUtility < 0.1) {
				relSqRootOfAvgUtility = 0;
				double relCountUtilityInSmallSD = 0;
				for (Iterator g = bidsUtil.iterator(); g.hasNext();) {
					double bidUtil = (Double) g.next();

					if (bidUtil >= agentData.oppFirstBidUtility
							&& (bidUtil < (relAvgUtility - relStdevUtility)
									|| bidUtil > (relAvgUtility
											+ relStdevUtility))) {
						relSqRootOfAvgUtility += (bidUtil - relAvgUtility)
								* (bidUtil - relAvgUtility);
						relCountUtilityInSmallSD++;
					}
				}
				relStdevUtility = Math
						.sqrt(relSqRootOfAvgUtility / relCountUtilityInSmallSD);
			}
			agentData.relStdevUtility = relStdevUtility;

			if (relCountUtility < 51) {
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("relCountUtility <= 50");
				// }
				agentData.minThreshold = relAvgUtility;
			} else {
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("relCountUtility > 50");
				// System.out.println("RelAvg: " + relAvgUtility + "RelStd: " +
				// relStdevUtility);
				// }
				agentData.minThreshold = relAvgUtility + this.discountingFactor
						* relStdevUtility * this.reservationValue;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage() + " exception in method calMinU");
		}

		return agentData;
	}

	private void updateConcedeDegree_largeDF(AgentData agentData) {
		try {
			double i = 0;
			agentData = updateConcedeDegree_UpdateMinU(agentData, "Large");
			agentData = MeasureConcedePartOfOppBehaviour(agentData);

			agentData.concedePartOfdiscountingFactor = this.discountingFactor
					- 1;
			agentData.concedePartOfOppBehaviour = (((((agentData.relCountLowerBoundMid2
					/ agentData.relCountUpperBoundMid2)
					* agentData.slopeOfSlopeOfSessionMax2)
					- ((agentData.relCountLowerBoundMid1
							/ agentData.relCountUpperBoundMid1)
							* agentData.slopeOfSlopeOfSessionMax1))
					/ this.k) / agentData.relStdevUtility) - this.N;
			if (agentData.startConcede == true) {
				i = agentData.concedePartOfdiscountingFactor
						+ agentData.concedePartOfOppBehaviour;
				agentData.concedeTime = Math.exp(i);
				if (agentData.concedeTime > 1) {
					agentData.concedeTime = 1;
				}
			} else {
				agentData.concedeTime = this.initialConcedeTime;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method updateConcedeDegree_largeDF");
		}
	}

	private AgentData MeasureConcedePartOfOppBehaviour(AgentData agentData) {
		try {
			agentData.count++;

			if (agentData.count < this.FirstTimeInterval) {
				if (utilitySpace.getUtility(
						agentData.oppPreviousBid) > agentData.utility_maximum_from_opponent_Session1) {
					agentData.utility_maximum_from_opponent_Session1 = utilitySpace
							.getUtility(agentData.oppPreviousBid);
				}
			} else if (agentData.count < this.SecondTimeInterval) {
				if (utilitySpace.getUtility(
						agentData.oppPreviousBid) > agentData.utility_maximum_from_opponent_Session2) {
					agentData.utility_maximum_from_opponent_Session2 = utilitySpace
							.getUtility(agentData.oppPreviousBid);
				}
			} else if (agentData.count < this.ThirdTimeInterval) {
				if (utilitySpace.getUtility(
						agentData.oppPreviousBid) > agentData.utility_maximum_from_opponent_Session3) {
					agentData.utility_maximum_from_opponent_Session3 = utilitySpace
							.getUtility(agentData.oppPreviousBid);
				}
			} else {
				agentData.relCountUpperBoundMid1 = 0;
				agentData.relCountUpperBoundMid2 = 0;
				agentData.relCountLowerBoundMid1 = 0;
				agentData.relCountLowerBoundMid2 = 0;
				agentData.midPointOfSlopeSessionMax1 = (agentData.utility_maximum_from_opponent_Session2
						+ agentData.utility_maximum_from_opponent_Session1) / 2;
				agentData.midPointOfSlopeSessionMax2 = (agentData.utility_maximum_from_opponent_Session3
						+ agentData.utility_maximum_from_opponent_Session2) / 2;
				agentData.slopeOfSlopeOfSessionMax1 = agentData.utility_maximum_from_opponent_Session2
						- agentData.utility_maximum_from_opponent_Session1;
				agentData.slopeOfSlopeOfSessionMax2 = agentData.utility_maximum_from_opponent_Session3
						- agentData.utility_maximum_from_opponent_Session2;

				LinkedList<Double> bidsUtil = new LinkedList();
				BidIterator myBidIterator = new BidIterator(
						this.utilitySpace.getDomain());
				for (; myBidIterator.hasNext();) {
					bidsUtil.add(
							this.utilitySpace.getUtility(myBidIterator.next()));
				}
				for (Iterator e = bidsUtil.iterator(); e.hasNext();) {
					double bidUtil = (Double) e.next();
					if (bidUtil >= agentData.midPointOfSlopeSessionMax1) {
						agentData.relCountUpperBoundMid1 += 1;
					} else {
						agentData.relCountLowerBoundMid1 += 1;
					}

					if (bidUtil >= agentData.midPointOfSlopeSessionMax2) {
						agentData.relCountUpperBoundMid2 += 1;
					} else {
						agentData.relCountLowerBoundMid2 += 1;
					}
				}
				agentData.utility_maximum_from_opponent_Session1 = agentData.utility_maximum_from_opponent_Session2;
				agentData.utility_maximum_from_opponent_Session2 = agentData.utility_maximum_from_opponent_Session3;
				agentData.utility_maximum_from_opponent_Session3 = 0;
				agentData.count = this.SecondTimeInterval - 1;
				agentData.startConcede = true;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method MeasureConcedePartOfOppBehaviour");
		}
		return agentData;
	}

	private AgentData updateConcedeDegree_nonDF(AgentData agentData) {
		try {
			if (agentData.IsOppFirstBid) {
				updateConcedeDegree_UpdateMinU(agentData, "Non");
				agentData.IsOppFirstBid = false;
			}

			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("Returned minThreshold:" +
			// agentData.minThreshold);
			// }
			// agentData.concedeTime = 0.85;
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method updateConcedeDegree_nonDF");
		}

		return agentData;
	}

	private LinkedList<Bid> SortBidsList(LinkedList<Bid> Bids) {
		LinkedList<Bid> sortedBidsList = new LinkedList<Bid>();
		try {
			HashMap<Bid, Double> sortedBidsMap = new HashMap<Bid, Double>();

			for (Bid bid : Bids) {
				sortedBidsMap.put(bid, this.utilitySpace.getUtility(bid));
			}

			List<Map.Entry<Bid, Double>> list_Data = new LinkedList<Map.Entry<Bid, Double>>(
					sortedBidsMap.entrySet());

			// Sorting
			Collections.sort(list_Data,
					new Comparator<Map.Entry<Bid, Double>>() {
						@Override
						public int compare(Map.Entry<Bid, Double> entry1,
								Map.Entry<Bid, Double> entry2) {
							return (entry1.getValue()
									.compareTo(entry2.getValue()));
						}
					});

			for (Map.Entry<Bid, Double> entry : list_Data) {
				sortedBidsList.add(entry.getKey());
			}
		} catch (Exception e) {
			System.out.println(
					e.getMessage() + " exception in method SortBidsList");
		}

		return sortedBidsList;
	}

	private LinkedList<Bid> SortBidsList(LinkedList<Bid> Bids,
			LinkedList<Double> distList) {
		LinkedList<Bid> sortedBidsList = new LinkedList<Bid>();
		try {
			HashMap<Bid, Double> sortedBidsMap = new HashMap<Bid, Double>();

			for (int i = 0; i < Bids.size(); i++) {
				sortedBidsMap.put(Bids.get(i), distList.get(i));
			}

			List<Map.Entry<Bid, Double>> list_Data = new LinkedList<Map.Entry<Bid, Double>>(
					sortedBidsMap.entrySet());

			// Sorting
			Collections.sort(list_Data,
					new Comparator<Map.Entry<Bid, Double>>() {
						@Override
						public int compare(Map.Entry<Bid, Double> entry1,
								Map.Entry<Bid, Double> entry2) {
							return (entry1.getValue()
									.compareTo(entry2.getValue()));
						}
					});

			for (Map.Entry<Bid, Double> entry : list_Data) {
				sortedBidsList.add(entry.getKey());
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method SortBidsList (Distance)");
		}

		return sortedBidsList;
	}

	private double CalWhetherOppFirstBidGoodEnough(double minThreshold,
			double oppFirstBidUtility) {
		try {
			LinkedList<Bid> Bids = new LinkedList<Bid>();
			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			for (; myBidIterator.hasNext();) {
				Bids.add(myBidIterator.next());
			}
			LinkedList<Bid> sortedBidsList = SortBidsList(Bids);
			double relSumUtility = 0;
			double relCountUtility = 0;
			for (int i = sortedBidsList.size() / 2; i < sortedBidsList
					.size(); i++) {
				relSumUtility += this.utilitySpace
						.getUtility(sortedBidsList.get(i));
				relCountUtility += 1;
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Selected Bid Utility:" +
				// this.utilitySpace.getUtility(sortedBidsList.get(i)));
				// }
			}

			double relAvgUtility = relSumUtility / relCountUtility;
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("Selected Bids Avg Utility:" + relAvgUtility +
			// ", oppFirstBidUtility:" + oppFirstBidUtility);
			// }
			if (oppFirstBidUtility >= relAvgUtility) {
				return oppFirstBidUtility;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method IsOppFirstBidGoodEnough");
		}

		return minThreshold;
	}

	private double Compare_MinThreshold_And_SecondMax(double minThreshold,
			double utility_SecondMaximum) {
		try {
			if (minThreshold > utility_SecondMaximum) {
				return (utility_SecondMaximum * 0.9);
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "exception in method Compare_MinThreshold_And_SecondMax");
		}

		return minThreshold;
	}

	private void PrintAllBids() {
		try {
			LinkedList<Bid> bids = new LinkedList();

			BidIterator myBidIterator = new BidIterator(
					this.utilitySpace.getDomain());
			for (; myBidIterator.hasNext();) {
				try {
					bids.add(myBidIterator.next());
				} catch (Exception e) {
					System.out.println(e.getMessage()
							+ "exception in method PrintAllBids");
				}
			}

			for (Iterator e = bids.iterator(); e.hasNext();) {
				Bid bid = (Bid) e.next();

				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println(Double.toString(this.utilitySpace.getUtility(bid))
				// + "\n");
				// }

				for (Issue lIssue : this.Issues) {
					int issueNum = lIssue.getNumber();
					Value v = bid.getValue(issueNum);

					if (debug) {
						file = new PrintStream(
								new FileOutputStream(fileName, true));
						System.setOut(file);
						System.out.print(v.toString() + ",");
					}
				}

				if (debug) {
					file = new PrintStream(
							new FileOutputStream(fileName, true));
					System.setOut(file);
					System.out.println();
				}
			}
		} catch (Exception e) {
			System.out.println(
					e.getMessage() + "exception in method PrintAllBids");
		}
	}

	private void printBidAllValues(Bid bid) {
		try {
			for (Issue lIssue : this.Issues) {
				int issueNum = lIssue.getNumber();
				Value v = bid.getValue(issueNum);

				if (debug) {
					file = new PrintStream(
							new FileOutputStream(fileName, true));
					System.setOut(file);
					System.out.print(v.toString() + "\t");
				}
			}

			if (debug) {
				file = new PrintStream(new FileOutputStream(fileName, true));
				System.setOut(file);
				System.out.println();
			}
		} catch (Exception e) {
			System.out.println(
					"Exception in printAllBidValues(): " + e.getMessage());
		}
	}

	private double CompareAdaptiveMinUAndNormalMinU(double adaptiveMinU,
			double normalMinU, double timeControl) {
		if (adaptiveMinU > normalMinU) {
			return (normalMinU + (adaptiveMinU - normalMinU) * timeControl);
		}
		return normalMinU;
	}

	private Bid BidToOffer_original() {
		Bid bidReturned = null;

		try {
			if (this.agentData1.minThreshold <= this.agentData2.minThreshold) {
				this.agentData2.minThreshold = this.agentData1.minThreshold;
			} else {
				this.agentData1.minThreshold = this.agentData2.minThreshold;
			}

			this.adaptiveMinUThreshold = this.offeredBidsAvg
					- this.offeredBidsSD * (1 - this.discountingFactor);
			// if(debug)
			// {
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println("calAdaptiveMinU:" + (this.offeredBidsAvg -
			// this.offeredBidsSD * (1 - this.discountingFactor)) +
			// ", AdaptiveAvg: " + this.offeredBidsAvg + ", AdaptiveSD:" +
			// this.offeredBidsSD + ", controlSD" + (this.offeredBidsSD * (1 -
			// this.discountingFactor)) + ", (1 - DF):" + (1 -
			// this.discountingFactor));
			// }
			double currentTIme = timeline.getTime();
			this.avgConcedeTime = (this.agentData2.concedeTime
					+ this.agentData1.concedeTime) / 2;
			double timeControlAdaptiveMinU = currentTIme / this.avgConcedeTime;
			if (currentTIme <= this.avgConcedeTime) {
				if (this.discountingFactor <= 0.5) {
					this.agentData1.minThreshold = (this.MaximumUtility
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.agentData1.concedeTime);
					this.agentData2.minThreshold = (this.MaximumUtility
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.agentData2.concedeTime);

					double minThresholdAgent1 = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData1.minThreshold,
							timeControlAdaptiveMinU);
					double minThresholdAgent2 = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData2.minThreshold,
							timeControlAdaptiveMinU);

					this.finalMinUAgent1 = minThresholdAgent1;
					this.finalMinUAgent2 = minThresholdAgent2;
					this.agentData1.utilitythreshold = this.MaximumUtility
							- (this.MaximumUtility - minThresholdAgent1)
									* Math.pow(
											(currentTIme
													/ this.agentData1.concedeTime),
											alpha1);
					this.agentData2.utilitythreshold = this.MaximumUtility
							- (this.MaximumUtility - minThresholdAgent2)
									* Math.pow(
											(currentTIme
													/ this.agentData2.concedeTime),
											alpha1);
				} else {
					double minThresholdAgent1 = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData1.minThreshold,
							timeControlAdaptiveMinU);
					double minThresholdAgent2 = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData2.minThreshold,
							timeControlAdaptiveMinU);
					this.finalMinUAgent1 = minThresholdAgent1;
					this.finalMinUAgent2 = minThresholdAgent2;

					this.agentData1.utilitythreshold = minThresholdAgent1
							+ (this.MaximumUtility - minThresholdAgent1)
									* (1 - Math.sin((Math.PI / 2) * (currentTIme
											/ this.agentData1.concedeTime)));
					this.agentData2.utilitythreshold = minThresholdAgent2
							+ (this.MaximumUtility - minThresholdAgent2)
									* (1 - Math.sin((Math.PI / 2) * (currentTIme
											/ this.agentData2.concedeTime)));
				}
			} else {
				if (this.discountingFactor <= 0.5) {
					this.agentData1.utilitythreshold = (this.MaximumUtility
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor, currentTIme);
					this.agentData2.utilitythreshold = (this.MaximumUtility
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor, currentTIme);
				} else if (this.discountingFactor == 1) {
					this.agentData1.utilitythreshold = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData1.minThreshold, 1);
					this.agentData2.utilitythreshold = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData2.minThreshold, 1);
					this.finalMinUAgent1 = this.agentData1.utilitythreshold;
					this.finalMinUAgent2 = this.agentData2.utilitythreshold;
				} else {
					double minThresholdAgent1 = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData1.minThreshold,
							timeControlAdaptiveMinU);
					double minThresholdAgent2 = CompareAdaptiveMinUAndNormalMinU(
							this.adaptiveMinUThreshold,
							this.agentData2.minThreshold,
							timeControlAdaptiveMinU);

					this.agentData1.utilitythreshold = minThresholdAgent1
							+ (this.MaximumUtility - minThresholdAgent1)
									/ (1 - this.agentData1.concedeTime)
									* Math.pow(
											(currentTIme
													- this.agentData1.concedeTime),
											this.discountingFactor);
					this.agentData2.utilitythreshold = minThresholdAgent2
							+ (this.MaximumUtility - minThresholdAgent2)
									/ (1 - this.agentData2.concedeTime)
									* Math.pow(
											(currentTIme
													- this.agentData2.concedeTime),
											this.discountingFactor);
				}
			}
			this.avgUtilitythreshold = (this.agentData1.utilitythreshold
					+ this.agentData2.utilitythreshold) / 2;
			if (this.avgUtilitythreshold > this.MaximumUtility) {
				this.avgUtilitythreshold = this.MaximumUtility;
			}

			/*
			 * if(minimumOfBid < 0.9 && this.guessOpponentType == false){
			 * if(this.opponentBidHistory.getSize() <= 2){ this.opponentType =
			 * 1;//tough opponent alpha1 = 2; } else{ this.opponentType = 0;
			 * alpha1 = 4; } this.guessOpponentType = true;//we only guess the
			 * opponent type once here System.out.println("we guess the opponent
			 * type is "+this.opponentType); }
			 */

			bidReturned = this.regeneratebid(this.avgUtilitythreshold);
			if (bidReturned == null) {
				// System.out.println("no bid is searched warning");
				bidReturned = this.bid_maximum_utility;
			}
			UpdateOfferedBidsStat(this.utilitySpace.getUtility(bidReturned));
			// if(debug)
			// {
			// //SysOut = System.out;
			// file = new PrintStream(new FileOutputStream(fileName, true));
			// System.setOut(file);
			// System.out.println(Double.toString(bidReturned) + "\n");
			// //System.setOut(SysOut);
			// }
			MinAcceptCondition(bidReturned);
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method BidToOffer_original");
			return this.bid_maximum_utility;
		}
		// System.out.println("the current threshold is " +
		// this.utilitythreshold + " with the value of alpha1 is " + alpha1);
		return bidReturned;
	}

	private boolean OtherAcceptCondition(Bid oppPreviousBid, boolean IsAccept) {
		try {
			if (timeline
					.getTime() >= ((this.agentData1.concedeTime
							+ this.agentData2.concedeTime) / 2)
					&& this.utilitySpace.getUtility(
							oppPreviousBid) >= this.minUtilityUhreshold) {
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("OtherAcceptCondition: true, minU:" +
				// this.minUtilityUhreshold + ", bidReturned:" +
				// this.utilitySpace.getUtility(oppPreviousBid));
				// }
				return true;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ " exception in method OtherAcceptCondition");
			return IsAccept;
		}
		return IsAccept;
	}

	private void MinAcceptCondition(Bid bidReturned) {
		try {
			if (this.minUtilityUhreshold > this.utilitySpace
					.getUtility(bidReturned)) {
				this.minUtilityUhreshold = this.utilitySpace
						.getUtility(bidReturned);
				// if(debug)
				// {
				// file = new PrintStream(new FileOutputStream(fileName, true));
				// System.setOut(file);
				// System.out.println("Changed minU:" + this.minUtilityUhreshold
				// + ", bidReturned:" +
				// this.utilitySpace.getUtility(bidReturned));
				// }
			}
		} catch (Exception e) {
			System.out.println(
					e.getMessage() + " exception in method MinAcceptCondition");
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}
}
