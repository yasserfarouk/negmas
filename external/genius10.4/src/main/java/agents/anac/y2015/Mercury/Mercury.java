package agents.anac.y2015.Mercury;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.Timeline;
import genius.core.utility.AdditiveUtilitySpace;

public class Mercury extends AbstractNegotiationParty {

	private double totalTime;
	private Action ActionOfOpponent = null;
	private Object party = null;
	private Object myparty = null;
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
	private Bid maxBid = null;
	private int myRounds = 0;

	private int numberOfParty = -1;
	private double acceConThres = 0.9; // when obtainning an acceptance signal,
										// try to seal the agreement with
										// utility discount on it own
	// private double miniLevel = 0;
	private ArrayList<Object> myParOrder = new ArrayList<Object>();
	private boolean halfSucc = false; // if true, our offer has been accepted by
										// one agent

	// private double succDis = 0.9;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		myparty = getPartyId();

		try {
			random = new Random();
			maximumOfBid = this.utilitySpace.getDomain()
					.getNumberOfPossibleBids();
			ownBidHistory = new OwnBidHistory();
			opponentBidHistory = new OpponentBidHistory();
			bidsBetweenUtility = new ArrayList<ArrayList<Bid>>();
			this.bid_maximum_utility = utilitySpace.getMaxUtilityBid();
			this.utilitythreshold = utilitySpace
					.getUtility(bid_maximum_utility); // initial
														// utility
														// threshold
			this.MaximumUtility = this.utilitythreshold;
			this.timeLeftAfter = 0;
			this.timeLeftBefore = 0;
			this.totalTime = timeline.getTotalTime();
			this.maximumTimeOfOpponent = 0;
			this.maximumTimeOfOwn = 0;
			this.minConcedeToDiscountingFactor = 0.08;// 0.1;
			this.discountingFactor = 1;
			if (utilitySpace.getDiscountFactor() <= 1D
					&& utilitySpace.getDiscountFactor() > 0D) {
				this.discountingFactor = utilitySpace.getDiscountFactor();
			}
			this.chooseUtilityThreshold();
			this.calculateBidsBetweenUtility();
			this.chooseConcedeToDiscountingDegree();
			this.opponentBidHistory
					.initializeDataStructures(utilitySpace.getDomain());
			this.timeLeftAfter = timeline.getCurrentTime();
			this.concedeToOpponent = false;
			this.toughAgent = false;
			this.alpha1 = 2;
			this.reservationValue = 0;
			if (utilitySpace.getReservationValue() > 0) {
				this.reservationValue = utilitySpace.getReservationValue();
			}

			maxBid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			System.out.println("initialization error" + e.getMessage());
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {
		super.receiveMessage(sender, opponentAction);

		this.ActionOfOpponent = opponentAction;
		this.party = sender;

		// System.out.println("party id is" +myparty+","+sender);

		if (numberOfParty == -1) {
			numberOfParty = getNumberOfParties();
			opponentBidHistory.initiPartyRec(numberOfParty,
					(AdditiveUtilitySpace) this.utilitySpace);
			// System.out.println("first informed
			// action"+opponentAction.toString());
		} else if (opponentAction instanceof Offer) {

			if (myParOrder.size() < numberOfParty)
				myParOrder.add(sender);

			this.opponentBidHistory.updateOpponentModel(
					((Offer) ActionOfOpponent).getBid(), sender,
					utilitySpace.getDomain(),
					(AdditiveUtilitySpace) this.utilitySpace);

			this.opponentBidHistory
					.updateOppRec(((Offer) ActionOfOpponent).getBid(), sender);
			// System.out.println("following informed
			// action"+opponentAction.toString());

		} else if (opponentAction instanceof Accept) {
			//
			// System.out.println("current informed action is
			// accept"+opponentAction.toString());
			// System.out.println("current informed action is
			// accept"+myParOrder.indexOf(sender)+","+myParOrder.indexOf(myparty));
			if (myParOrder.size() < numberOfParty)
				myParOrder.add(sender);

			if ((myParOrder.indexOf(sender) + 1) % numberOfParty == myParOrder
					.indexOf(myparty)
					|| (myParOrder.indexOf(sender) - 1)
							% numberOfParty == myParOrder.indexOf(myparty)) {

				halfSucc = true; // true
				System.out.println("Half success action caused by " + sender
						+ "," + opponentAction.toString());

				this.opponentBidHistory.updateOpponentModel(
						ownBidHistory.getLastBid(), sender,
						utilitySpace.getDomain(),
						(AdditiveUtilitySpace) this.utilitySpace);

				this.opponentBidHistory.updateAccRec(ownBidHistory.getLastBid(),
						sender);

			} else {
				// only works when agents are less than 3!
				this.opponentBidHistory.updateOpponentModel(
						opponentBidHistory.getLastOppBid(), sender,
						utilitySpace.getDomain(),
						(AdditiveUtilitySpace) this.utilitySpace);

				this.opponentBidHistory.updateAccRec(
						opponentBidHistory.getLastOppBid(), sender);
			}

		} else {
			//
			if (myParOrder.size() < numberOfParty)
				myParOrder.add(sender);

			System.out.println("Unexpected action from " + sender + ","
					+ opponentAction.toString());
		}

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		Action action = null;
		Bid bid = null;
		myRounds++;

		// System.out.println("halfSucces " +
		// halfSucc+","+myRounds+","+this.estimateRoundLeft(false)+","+estimateRoundLeft(true));

		if (myParOrder.size() < numberOfParty)
			myParOrder.add(myparty);

		try {
			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());
			this.timeLeftBefore = timeline.getCurrentTime();

			// we propose first and propose the bid with maximum utility
			if (!validActions.contains(Accept.class)) {
				bid = this.bid_maximum_utility;
				action = new Offer(getPartyId(), bid);
			} else if (ActionOfOpponent instanceof Offer) {// the opponent
															// propose first and
															// we response
															// secondly
				// update opponent model first
				// this.opponentBidHistory.updateOpponentModel(
				// ((Offer) ActionOfOpponent).getBid(),
				// utilitySpace.getDomain(), this.utilitySpace);

				this.updateConcedeDegree();
				// update the estimation
				if (myRounds <= 30) {
					// bid = utilitySpace.getMaxUtilityBid();
					bid = this.bid_maximum_utility;

					action = new Offer(getPartyId(), bid);
				} else {// other conditions
					if (timeline.getTime() < 0.97) {// still have some rounds
													// left to further
													// negotiate (the major
													// negotiation period)
						if (!this.halfSucc) {
							bid = BidToOffer();
						} else {
							bid = NiceBidtoOffer();
						}

						Boolean IsAccept = AcceptOpponentOffer(
								((Offer) ActionOfOpponent).getBid(), bid);
						Boolean IsTerminate = TerminateCurrentNegotiation(bid);
						if (IsAccept && !IsTerminate) {
							action = new Accept(getPartyId(),
									this.opponentBidHistory.getLastOppBid());
						} else if (IsTerminate && !IsAccept) {
							action = new EndNegotiation(getPartyId());
							// action = new Offer(maxBid);
						} else if (IsAccept && IsTerminate) {
							if (this.utilitySpace
									.getUtility(((Offer) ActionOfOpponent)
											.getBid()) > this.reservationValue) {
								action = new Accept(getPartyId(),
										this.opponentBidHistory
												.getLastOppBid());
							} else {
								action = new EndNegotiation(getPartyId());
								// action = new Offer(maxBid);
							}
						} else {
							// we expect that the negotiation is over once we
							// select a bid from the opponent's history.
							if (this.concedeToOpponent == true) {
								// bid =
								// opponentBidHistory.chooseBestFromHistory(this.utilitySpace);
								bid = opponentBidHistory.getBestBidInHistory();

								action = new Offer(getPartyId(), bid);
								this.toughAgent = true;
								this.concedeToOpponent = false;
							} else {
								action = new Offer(getPartyId(), bid);
								this.toughAgent = false;
							}
						}
					} else {// this is the last chance and we concede by
							// providing the opponent the best offer he ever
							// proposed to us
						// in this case, it corresponds to an opponent whose
						// decision time is short
						if (timeline.getTime() > 0.995) {
							// bid =
							// opponentBidHistory.chooseBestFromHistory(this.utilitySpace);

							if (!this.halfSucc) {
								bid = opponentBidHistory.getBestBidInHistory();
							} else {
								bid = NiceBidtoOffer();
							}

							// this is specially designed to avoid that we got
							// very low utility by searching between an
							// acceptable range (when the domain is small)
							if (this.utilitySpace.getUtility(bid) < 0.75) {
								List<Bid> candidateBids = this
										.getBidsBetweenUtility(
												this.MaximumUtility - 0.15,
												this.MaximumUtility - 0.02);
								// if the candidate bids do not exsit and also
								// the deadline is approaching in next round, we
								// concede.
								// if (candidateBids.size() == 1 &&
								// timeline.getTime()>0.9998) {
								// we have no chance to make a new proposal
								// before the deadline
								if (timeline.getTime() > 0.9999) {
									// bid =
									// opponentBidHistory.getMiniBestOpp(); //
									// more concessive

									if (this.utilitySpace
											.getUtility(opponentBidHistory
													.getMiniBestOpp()) > 0.5) {
										bid = opponentBidHistory
												.getMiniBestOpp(); // more
																	// concessive
									} else if (this.utilitySpace
											.getUtility(opponentBidHistory
													.getBestBidInHistory()) > 0.5) {
										bid = opponentBidHistory
												.getBestBidInHistory();
									} else {
										bid = ownBidHistory.getLastBid();
									}

								} else {
									bid = opponentBidHistory.ChooseBid(
											candidateBids,
											this.utilitySpace.getDomain());
								}
								if (bid == null) {
									// bid = opponentBidHistory
									// .getBestBidInHistory();
									bid = ownBidHistory.getLastBid();
								}
							}
							Boolean IsAccept = AcceptOpponentOffer(
									((Offer) ActionOfOpponent).getBid(), bid);
							Boolean IsTerminate = TerminateCurrentNegotiation(
									bid);
							if (IsAccept && !IsTerminate) {
								action = new Accept(getPartyId(),
										this.opponentBidHistory
												.getLastOppBid());
							} else if (IsTerminate && !IsAccept) {
								action = new EndNegotiation(getPartyId());
								// action = new Offer(maxBid);
							} else if (IsTerminate && IsAccept) {
								if (this.utilitySpace
										.getUtility(((Offer) ActionOfOpponent)
												.getBid()) > this.reservationValue) {
									action = new Accept(getPartyId(),
											this.opponentBidHistory
													.getLastOppBid());
								} else {
									action = new EndNegotiation(getPartyId());
									// action = new Offer(maxBid);
								}
							} else {
								if (this.toughAgent == true) {
									action = new Accept(getPartyId(),
											this.opponentBidHistory
													.getLastOppBid());
								} else {
									action = new Offer(getPartyId(), bid);
								}
							}
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
							if (!this.halfSucc) {
								bid = opponentBidHistory.getBestBidInHistory();
							} else {
								bid = NiceBidtoOffer();
							}

							// System.out.println("test----------------------------------------------------------"
							// + timeline.getTime());
							Boolean IsAccept = AcceptOpponentOffer(
									((Offer) ActionOfOpponent).getBid(), bid);
							Boolean IsTerminate = TerminateCurrentNegotiation(
									bid);

							if (IsAccept && !IsTerminate) {
								action = new Accept(getPartyId(),
										this.opponentBidHistory
												.getLastOppBid());
							} else if (IsTerminate && !IsAccept) {
								action = new EndNegotiation(getPartyId());
								// action = new Offer(maxBid);
							} else if (IsAccept && IsTerminate) {
								if (this.utilitySpace
										.getUtility(((Offer) ActionOfOpponent)
												.getBid()) > this.reservationValue) {
									action = new Accept(getPartyId(),
											this.opponentBidHistory
													.getLastOppBid());
								} else {
									action = new EndNegotiation(getPartyId());
									// action = new Offer(maxBid);
								}
							} else {
								action = new Offer(getPartyId(), bid);

							}
						}
					}
				}

				if (myRounds > 2) {
					if (ownBidHistory.isInsideMyBids(
							((Offer) ActionOfOpponent).getBid()))
						action = new Accept(getPartyId(),
								this.opponentBidHistory.getLastOppBid());
				}

			} else if (ActionOfOpponent instanceof Accept) {

				// bid = opponentBidHistory.getLastAcce();

				/*
				 * try{
				 * 
				 * if ( this.utilitythreshold*acceConThres <=
				 * this.utilitySpace.getUtility
				 * (opponentBidHistory.getLastOppBid()) ){ System.out.println(
				 * "accept an offer after seeing an acceptance action with a not bad result"
				 * +ActionOfOpponent.getAgent()); action = new Accept(); }else{
				 * 
				 * try{
				 * 
				 * this.utilitythreshold =
				 * utilitySpace.getUtility(ownBidHistory.getLastBid()); double
				 * max = this.utilitythreshold * 1.2; if(max >
				 * this.MaximumUtility) max = this.MaximumUtility; //double min
				 * = (this.utilitythreshold -
				 * utilitySpace.getUtility(ownBidHistory.getLastBid())) +
				 * utilitySpace.getUtility(ownBidHistory.getLastBid()); double
				 * min = this.utilitythreshold * 0.95 * timeline.getTime();
				 * //System.out.println("test in method NiceBidtoOffer"+this.
				 * utilitythreshold+","+max+","+min); if (min < 0.5){ min = 0.5;
				 * if(min > max) max = min; }
				 * 
				 * List<Bid> candidateBids = getBidsBetweenUtility(min, max);
				 * 
				 * Random random = new Random();
				 * 
				 * int s =0;
				 * 
				 * try{ s = random.nextInt(candidateBids.size()); }catch
				 * (Exception e){ s = 0; }
				 * 
				 * bid = candidateBids.get(s);
				 * 
				 * }catch (Exception e){ System.out.println("error - 1"); bid =
				 * ownBidHistory.getLastBid(); }
				 * 
				 * 
				 * } //action = new Offer(ownBidHistory.getLastBid()); }catch
				 * (Exception ee) { action = new Accept(); }
				 */

				try {

					if (this.utilitythreshold
							* acceConThres <= this.utilitySpace.getUtility(
									opponentBidHistory.getLastOppBid())) {
						System.out.println(
								"accept an offer after seeing an acceptance action with a not bad result"
										+ ActionOfOpponent.getAgent());
						action = new Accept(getPartyId(),
								this.opponentBidHistory.getLastOppBid());
					} else {
						bid = BidToOffer();

						Boolean IsAccept = AcceptOpponentOffer(
								((Offer) ActionOfOpponent).getBid(), bid);
						Boolean IsTerminate = TerminateCurrentNegotiation(bid);

						if (IsAccept && !IsTerminate) {
							action = new Accept(getPartyId(),
									this.opponentBidHistory.getLastOppBid());
						} else if (IsTerminate && !IsAccept) {
							action = new EndNegotiation(getPartyId());
							// action = new Offer(maxBid);
						} else if (IsAccept && IsTerminate) {
							if (this.utilitySpace
									.getUtility(((Offer) ActionOfOpponent)
											.getBid()) > this.reservationValue) {
								action = new Accept(getPartyId(),
										this.opponentBidHistory
												.getLastOppBid());
							} else {
								action = new EndNegotiation(getPartyId());
								// action = new Offer(maxBid);
							}
						} else {
							action = new Offer(getPartyId(), bid);
						}

					}

				} catch (Exception e) {

				}

			}
			// System.out.println("i propose " + debug + " bid at time " +
			// timeline.getTime());

			// System.out.println("exception number - 3, Ares!"+bid);

			// if (bid != null && !ownBidHistory.isInsideMyBids(bid))
			// this.ownBidHistory.addBid(bid, utilitySpace);

			if (bid != null)
				this.ownBidHistory.addBid(bid,
						(AdditiveUtilitySpace) utilitySpace);

			this.timeLeftAfter = timeline.getCurrentTime();
			this.estimateRoundLeft(false);// update the estimation
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			System.out.println(estimateRoundLeft(false));
			// action = new Accept(getAgentID()); // accept if anything goes
			if (ActionOfOpponent instanceof Accept) {
				action = new Accept(getPartyId(),
						this.opponentBidHistory.getLastOppBid());
			} else {
				System.out.println("handle exception - 1");
				// action = new EndNegotiation();
			}
			// action = new EndNegotiation(); // terminate if anything
			// action = new Offer(maxBid); // goes wrong.
		}

		if (action == null)
			action = new Offer(getPartyId(), maxBid);

		if (this.discountingFactor <= 0.5 && this.reservationValue >= 0.45
				&& timeline.getTime() > 0.15)
			action = new EndNegotiation(getPartyId());

		if (timeline.getCurrentTime() > timeline.getTotalTime() * 1.1) {
			System.out.println("exception in negotiation time for Ares!"
					+ timeline.getCurrentTime() + ","
					+ timeline.getTotalTime());
			action = new EndNegotiation(getPartyId());
			// return action;
		}

		/*
		 * if (ownBidHistory.numOfBidsProposed() > 30){
		 * 
		 * System.out.println("exception number - 2, Ares!"+opponentBidHistory.
		 * partyOrder .size()+","+opponentBidHistory.partyOrder.get(0)+","+
		 * opponentBidHistory .partyOrder.get(1));
		 * 
		 * try{ System.out.println("max opp utili, Ares!"+this.utilitySpace.
		 * getUtility (this.opponentBidHistory.getBestBidInHistory())+","
		 * +this.utilitySpace
		 * .getUtility(this.opponentBidHistory.getBestOpp(this.
		 * opponentBidHistory.partyOrder.get(0)))+","
		 * +this.utilitySpace.getUtility
		 * (this.opponentBidHistory.getBestOpp(this.
		 * opponentBidHistory.partyOrder.get(1)))); } catch (Exception e){
		 * System.out.println("exception number - 1, Ares!"); } }
		 */

		return action;
	}

	/*
	 * principle: randomization over those candidate bids to let the opponent
	 * have a better model of my utility profile return the bid to be offered in
	 * the next round
	 */
	private Bid BidToOffer() {
		Bid bidReturned = null;
		double decreasingAmount_1 = 0.09; // 0.05
		double decreasingAmount_2 = 0.25;
		try {

			double maximumOfBid = this.MaximumUtility;// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
			double minimumOfBid = this.MaximumUtility;
			// used when the domain is very large.
			// make concession when the domin is large
			if (this.discountingFactor == 1 && this.maximumOfBid > 3200) {
				// minimumOfBid = this.MaximumUtility - decreasingAmount_1;

				/*
				 * if (timeline.getTime() <= 0.68){ minimumOfBid =
				 * this.MaximumUtility * (1 - decreasingAmount_1 *
				 * Math.pow(timeline.getTime(),3)); }else if(timeline.getTime()
				 * > 0.68 && timeline.getTime() <= 0.81 ){ minimumOfBid =
				 * this.MaximumUtility * (1 -
				 * decreasingAmount_1*1.3*Math.pow(timeline.getTime(),2)); }else
				 * if (timeline.getTime() > 0.81 ){ minimumOfBid =
				 * this.MaximumUtility * (1 -
				 * decreasingAmount_1*(0.4+timeline.getTime())); }
				 */

				minimumOfBid = this.MaximumUtility * 0.94
						+ 0.06 * this.MaximumUtility
								* (1 - Math.pow(timeline.getTime(), 3));

				// make further concession when the deadline is approaching and
				// the domain is large
				if (this.discountingFactor > 1 - decreasingAmount_2
						&& this.maximumOfBid > 10000
						&& timeline.getTime() >= 0.98) {
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
				if (timeline.getTime() <= this.concedeToDiscountingFactor) {
					double minThreshold = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									this.concedeToDiscountingFactor);
					this.utilitythreshold = maximumOfBid
							- (maximumOfBid - minThreshold) * Math.pow(
									(timeline.getTime()
											/ this.concedeToDiscountingFactor),
									alpha1);
				} else {
					this.utilitythreshold = (maximumOfBid
							* this.discountingFactor)
							/ Math.pow(this.discountingFactor,
									timeline.getTime());
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

			/*
			 * if (halfSucc){ minimumOfBid = minimumOfBid * succDis *
			 * timeline.getTime(); this.utilitythreshold = this.utilitythreshold
			 * * succDis * timeline.getTime(); }
			 */

			// Bid bestBidOfferedByOpponent =
			// opponentBidHistory.getBestBidInHistory();
			/*
			 * if (halfSucc){
			 * if(utilitySpace.getUtility(opponentBidHistory.getMiniBestOpp()) >
			 * 0.51){ minimumOfBid =
			 * utilitySpace.getUtility(opponentBidHistory.getMiniBestOpp
			 * ())*(2-Math.pow(timeline.getTime(),2.5)); }else{ minimumOfBid =
			 * 0.51*(2-Math.pow(timeline.getTime(),2.5)); }
			 * 
			 * maximumOfBid =
			 * utilitySpace.getUtility(opponentBidHistory.getLastAcce());
			 * 
			 * if (maximumOfBid < minimumOfBid){ maximumOfBid =
			 * minimumOfBid*1.05; } }
			 */

			Bid bestBidOfferedByOpponent = opponentBidHistory.getMiniBestOpp();

			if (utilitySpace.getUtility(
					bestBidOfferedByOpponent) >= this.utilitythreshold
					|| utilitySpace.getUtility(
							bestBidOfferedByOpponent) >= minimumOfBid) {
				return bestBidOfferedByOpponent;
			}
			List<Bid> candidateBids = this.getBidsBetweenUtility(minimumOfBid,
					maximumOfBid);

			bidReturned = opponentBidHistory.ChooseBid(candidateBids,
					this.utilitySpace.getDomain());

			if (bidReturned == null) {
				System.out.println("no bids can be found.");
				bidReturned = this.utilitySpace.getMaxUtilityBid();
			}
		} catch (Exception e) {
			System.out
					.println(e.getMessage() + "exception in method BidToOffer");
		}
		// System.out.println("the current threshold is " +
		// this.utilitythreshold + " with the value of alpha1 is " + alpha1);
		return bidReturned;
	}

	private Bid NiceBidtoOffer() {
		Bid bid = null;

		try {

			// this.utilitythreshold =
			// utilitySpace.getUtility(opponentBidHistory.getLastAcce());
			double max = this.utilitythreshold * 1.05;
			// double min =
			// utilitySpace.getUtility(opponentBidHistory.getMiniBestOpp());
			double min = this.utilitythreshold * 0.95 * timeline.getTime();
			// System.out.println("test in method
			// NiceBidtoOffer"+this.utilitythreshold+","+max+","+min);

			if (min < 0.5) {
				min = 0.5;
			}
			if (max < 0.5)
				max = 0.5;

			bid = genRanBid(min, max);

		} catch (Exception e) {
			System.out.println("error in method NiceBidtoOffer");
			bid = ownBidHistory.getLastBid();
		}

		return bid;
	}

	/*
	 * decide whether to accept the current offer or not
	 */
	private boolean AcceptOpponentOffer(Bid opponentBid, Bid ownBid) {
		double currentUtility = 0;
		double nextRoundUtility = 0;
		double maximumUtility = 0;
		this.concedeToOpponent = false;
		try {
			currentUtility = this.utilitySpace.getUtility(opponentBid);
			maximumUtility = this.MaximumUtility;// utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "Exception in method AcceptOpponentOffer part 1");
		}
		try {
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "Exception in method AcceptOpponentOffer part 2");
		}
		// System.out.println(this.utilitythreshold +"at time "+
		// timeline.getTime());
		if (currentUtility >= this.utilitythreshold
				|| currentUtility >= nextRoundUtility) {
			return true;
		} else {
			// if the current utility with discount is larger than the predicted
			// maximum utility with discount
			// then accept it.
			double predictMaximumUtility = maximumUtility
					* this.discountingFactor;
			// double currentMaximumUtility =
			// this.utilitySpace.getUtilityWithDiscount(opponentBidHistory.chooseBestFromHistory(utilitySpace),
			// timeline);
			double currentMaximumUtility = this.utilitySpace
					.getUtilityWithDiscount(
							opponentBidHistory.getBestBidInHistory(), timeline);
			if (currentMaximumUtility > predictMaximumUtility
					&& timeline.getTime() > this.concedeToDiscountingFactor) {
				try {
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace
							.getUtility(opponentBid) >= utilitySpace.getUtility(
									opponentBidHistory.getBestBidInHistory())
									- 0.01) {
						return true;
					} else {
						this.concedeToOpponent = true;
						return false;
					}
				} catch (Exception e) {
					System.out
							.println("exception in Method AcceptOpponentOffer");
					return true;
				}
				// retrieve the opponent's biding history and utilize it
			} else if (currentMaximumUtility > this.utilitythreshold
					* Math.pow(this.discountingFactor, timeline.getTime())) {
				try {
					// if the current offer is approximately as good as the best
					// one in the history, then accept it.
					if (utilitySpace
							.getUtility(opponentBid) >= utilitySpace.getUtility(
									opponentBidHistory.getBestBidInHistory())
									- 0.01) {
						return true;
					} else {
						System.out.println(
								"test" + utilitySpace.getUtility(opponentBid)
										+ this.utilitythreshold);
						this.concedeToOpponent = true;
						return false;
					}
				} catch (Exception e) {
					System.out
							.println("exception in Method AcceptOpponentOffer");
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
		this.concedeToOpponent = false;
		try {
			currentUtility = this.reservationValue;
			nextRoundUtility = this.utilitySpace.getUtility(ownBid);
			maximumUtility = this.MaximumUtility;
		} catch (Exception e) {
			System.out.println(e.getMessage()
					+ "Exception in method TerminateCurrentNegotiation part 1");
		}

		if (currentUtility >= this.utilitythreshold
				|| currentUtility >= nextRoundUtility) {
			return true;
		} else {
			// if the current reseravation utility with discount is larger than
			// the predicted maximum utility with discount
			// then terminate the negotiation.
			double predictMaximumUtility = maximumUtility
					* this.discountingFactor;
			double currentMaximumUtility = this.utilitySpace
					.getReservationValueWithDiscount(timeline);
			// System.out.println("the current reserved value is "+
			// this.reservationValue+" after discounting is
			// "+currentMaximumUtility);
			if (currentMaximumUtility > predictMaximumUtility
					&& timeline.getTime() > this.concedeToDiscountingFactor) {
				return true;
			} else {
				return false;
			}
		}
	}

	/*
	 * estimate the number of rounds left before reaching the deadline @param
	 * opponent @return
	 */

	private int estimateRoundLeft(boolean opponent) {
		double round;

		if (timeline.getType() == Timeline.Type.Rounds) {

			// System.out.println((int)(timeline.getTotalTime() -
			// timeline.getCurrentTime()));

			return (int) (timeline.getTotalTime() - timeline.getCurrentTime());
		} else {

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
			// timeline.getElapsedSeconds()
			// + "---" + round + "----" + this.maximumTimeOfOpponent);
			return ((int) (round));
		}
	}

	/*
	 * pre-processing to save the computational time each round
	 */
	private void calculateBidsBetweenUtility() {
		BidIterator myBidIterator = new BidIterator(
				this.utilitySpace.getDomain());

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
			this.bidsBetweenUtility.get(maximumRounds - 1)
					.add(this.bid_maximum_utility);
			// note that here we may need to use some trick to reduce the
			// computation cost (to be checked later);
			// add those bids in each range into the corresponding arraylist
			int limits = 0;
			if (this.maximumOfBid < 20000) {
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
			} else {
				while (limits <= 20000) {
					Bid b = this.RandomSearchBid();
					for (int i = 0; i < maximumRounds; i++) {
						if (utilitySpace.getUtility(b) <= (i + 1) * 0.01
								+ minUtility
								&& utilitySpace.getUtility(b) >= i * 0.01
										+ minUtility) {
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
			beta = 1.85;
		} else if (this.discountingFactor > 0.5) {
			beta = 1.63;
		} else {
			beta = 1.2;
		}
		alpha = Math.pow(this.discountingFactor, beta);
		this.concedeToDiscountingFactor = this.minConcedeToDiscountingFactor
				+ (1 - this.minConcedeToDiscountingFactor) * alpha;
		this.concedeToDiscountingFactor_original = this.concedeToDiscountingFactor;
		// System.out.println("concedeToDiscountingFactor is "
		// + this.concedeToDiscountingFactor + "current time is "
		// + timeline.getTime());
	}

	/*
	 * update the concede-to-time degree based on the predicted toughness degree
	 * of the opponent
	 */

	private void updateConcedeDegree() {
		double gama = 10;
		double weight = 0.1;
		double opponnetToughnessDegree = this.opponentBidHistory
				.getConcessionDegree();
		// this.concedeToDiscountingFactor =
		// this.concedeToDiscountingFactor_original * (1 +
		// opponnetToughnessDegree);
		this.concedeToDiscountingFactor = this.concedeToDiscountingFactor_original
				+ weight * (1 - this.concedeToDiscountingFactor_original)
						* Math.pow(opponnetToughnessDegree, gama);
		if (this.concedeToDiscountingFactor >= 1) {
			this.concedeToDiscountingFactor = 1;
		}
		// System.out.println("concedeToDiscountingFactor is " +
		// this.concedeToDiscountingFactor + "current time is " +
		// timeline.getTime() + "original concedetodiscoutingfactor is " +
		// this.concedeToDiscountingFactor_original);
	}

	private Bid genRanBid(double min, double max) {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		int counter = 0;
		int limit = 1000;
		double fmax = max;

		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
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
					values.put(lIssueReal.getNumber(), new ValueReal(lIssueReal
							.getLowerBound()
							+ (lIssueReal.getUpperBound()
									- lIssueReal.getLowerBound()) * (optionInd)
									/ (lIssueReal
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
					//
				}
			}

			try {
				bid = new Bid(utilitySpace.getDomain(), values);
			} catch (Exception e) {
				// System.out.println("error in generating random bids");
			}

			counter++;
			if (counter > limit) {
				limit = limit + 500;
				fmax += 0.005;
				// return mBidHistory.getMyLastBid();
			}

			if (counter > 4000)
				return ownBidHistory.getLastBid();

		} while (getUtility(bid) < min || getUtility(bid) > fmax);

		return bid;
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
