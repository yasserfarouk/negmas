package agents.anac.y2014.AgentQuest;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.utility.AbstractUtilitySpace;

/**
 * 
 * @author Edwin Yaqub - (GWDG, Goettingen, Germany)
 * @contact edwinyaqub@yahoo.com
 * 
 */
public class AgentQuest extends Agent {

	private Action partnerAction;
	private AbstractUtilitySpace myUtilitySpace;
	private Bid myLastBid;
	private Bid bestBid;
	private Bid opponentLastBid;
	private int hardlineCounter = 0;
	private final int opponentMonitoringWindowSize = 10;
	private int opponentMonitoringWindowIndex = 0;
	private double opponentUtilityInWindowSofar = 0;
	private final int lastMonitoringWindowCarryOnHardHeadednessValue = 3;
	int counter = 0;
	private Domain domain;
	private BidHistory myBidHistory = new BidHistory();
	private BidHistory opponentBidHistory = new BidHistory();
	private int MAX_BIDS_TO_SAMPLE = 400;
	private int MAX_BIDS_TO_STORE = 1000;
	private int alpha = 4; // Acceptable threshold for opponent harheadedness:
							// adjust between 1 and opponentMonitoringWindowSize
							// to negotiate hard (low value) or soft (high
							// value).
	private double beta = 0.6d; // Acceptable threshold for conceding
								// probability: adjust between 0 and 1 to
								// negotiate hard (high value) or soft (low
								// value).
	private double defaultReservationValue = 0.25d;
	private double meanResponseTimeOfOpponent = 0.0d;
	private double lastResponseTime = 0.0d;

	@Override
	public void init() {
		System.out.println("\n" + getName() + " version(" + getVersion()
				+ ") initializing...");
		this.myUtilitySpace = super.utilitySpace;
		this.domain = this.myUtilitySpace.getDomain();
		if (this.myUtilitySpace.getReservationValue() != null
				&& this.myUtilitySpace.getReservationValue() > 0.0d) {
			this.defaultReservationValue = this.myUtilitySpace
					.getReservationValue().doubleValue();
		}
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		this.partnerAction = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			if (this.partnerAction == null) {
				// I am initiator - send Bid with (known) highest Utility
				double time = this.timeline.getTime();
				addSampleBids(time);
				addSampleBids(time); // Increase bid samples at our side so we
										// dont loose on a low utility due to
										// less samples in first round.
				this.myLastBid = this.myBidHistory.getBestBidDetails().getBid();
				action = new Offer(getAgentID(), this.myLastBid);
				this.counter++;
				this.lastResponseTime = time;
			}

			if (this.partnerAction instanceof Offer) {
				this.counter++;
				this.opponentMonitoringWindowIndex++;
				double time = this.timeline.getTime();

				// not the first round. measure mean response time and update
				// last round's time
				if (this.counter > 1) {
					this.meanResponseTimeOfOpponent += (time
							- this.lastResponseTime);
					this.meanResponseTimeOfOpponent /= 2;
					this.lastResponseTime = time;// reset lastResponseTime
				} else { // this is first round
					this.lastResponseTime = time;
					addSampleBids(time); // increase bid samples from our side
											// so we dont loose on a low utility
											// due to less samples in first
											// round.
				}

				addSampleBids(time);

				if (this.myLastBid == null) {
					this.myLastBid = this.myBidHistory.getBestBidDetails()
							.getBid();
				}

				// Received an Offer from Opponent - process it, generate
				// CounterOffer and decide whether Offer is acceptable or to
				// send CounterOffer
				Offer partnerOffer = (Offer) this.partnerAction;
				Bid partnerBid = partnerOffer.getBid();

				addOpponentBidToHistory(partnerBid, time);

				// Save the first partner Bid (happens just one time):
				if (this.opponentLastBid == null) {
					this.opponentLastBid = partnerBid;
					opponentUtilityInWindowSofar = this.myUtilitySpace
							.getUtilityWithDiscount(this.opponentLastBid, time);
				}

				if (this.bestBid == null) {
					// First bid from opponent (happens just one time)
					this.bestBid = partnerBid;
				}

				double offeredUtilFromPartner = this.myUtilitySpace
						.getUtilityWithDiscount(partnerBid, time);

				// Tolerance measure:
				if (this.opponentMonitoringWindowIndex > 1
						&& this.opponentMonitoringWindowIndex <= this.opponentMonitoringWindowSize) { // this.lastTenCounter
																										// >
																										// 0
																										// &&
					// Maintaining mean utility of opponent's offer with
					// opponent utility in window sofar:
					opponentUtilityInWindowSofar += offeredUtilFromPartner;
					opponentUtilityInWindowSofar /= 2;
					if (offeredUtilFromPartner <= opponentUtilityInWindowSofar) {
						if (this.hardlineCounter > this.opponentMonitoringWindowSize) {
							this.hardlineCounter = this.lastMonitoringWindowCarryOnHardHeadednessValue;
						} else {
							this.hardlineCounter++;
						}
					} else {
						if (this.hardlineCounter <= 0) {
							this.hardlineCounter = 0;
						} else {
							this.hardlineCounter--;
						}
					}
				}

				// Saving opponent's bid as lastBid:
				this.opponentLastBid = partnerBid;

				// Maintain best Bid:
				if (this.myUtilitySpace.getUtilityWithDiscount(partnerBid,
						time) > this.myUtilitySpace
								.getUtilityWithDiscount(this.bestBid, time)) {
					this.bestBid = partnerBid;
				}

				if (genearteCounterOffer(partnerOffer, time) == null) {
					// To deal with the unlikely case if we cant produce a
					// counter offer.
					action = new Accept(getAgentID(), opponentLastBid);
				} else {
					if (acceptOpponentOffer(partnerOffer, time)) {
						action = new Accept(getAgentID(), opponentLastBid);
					} else {
						action = new Offer(getAgentID(), this.myLastBid);
					}
				}
			}

			if (this.opponentMonitoringWindowIndex == this.opponentMonitoringWindowSize) {
				// reset the opponentMonitoringWindowIndex and hardlineCounter:
				this.opponentMonitoringWindowIndex = 0;
				if (this.hardlineCounter > this.opponentMonitoringWindowSize) {
					this.hardlineCounter = this.lastMonitoringWindowCarryOnHardHeadednessValue;
				}
				if (this.hardlineCounter < 0) {
					this.hardlineCounter = 0;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			action = new Accept(getAgentID(), opponentLastBid);
		}

		return action;
	}

	private void addOpponentBidToHistory(Bid partnerBid, double time) {
		if (this.opponentBidHistory.size() < this.MAX_BIDS_TO_STORE) {
			try {
				this.opponentBidHistory
						.add(new BidDetails(partnerBid, this.myUtilitySpace
								.getUtilityWithDiscount(partnerBid, time)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			// Make room by removing the worst 1 bid
			this.opponentBidHistory = new BidHistory(this.opponentBidHistory
					.getNBestBids(this.MAX_BIDS_TO_STORE - 1));
			try {
				this.opponentBidHistory
						.add(new BidDetails(partnerBid, this.myUtilitySpace
								.getUtilityWithDiscount(partnerBid, time)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		this.myBidHistory.sortToUtility();
	}

	private void addSampleBids(double time) {
		if (this.myBidHistory.size() < this.MAX_BIDS_TO_STORE) {
			addBids(time);
		} else {
			// Make room by removing the worst 100 bids
			this.myBidHistory = new BidHistory(
					this.myBidHistory.getNBestBids(900));
			addBids(time);
		}
	}

	private void addBids(double time) {
		for (int i = 0; i < this.MAX_BIDS_TO_SAMPLE; i++) {
			Bid randomBid = this.domain.getRandomBid(null);
			try {
				this.myBidHistory
						.add(new BidDetails(randomBid, this.myUtilitySpace
								.getUtilityWithDiscount(randomBid, time)));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		this.myBidHistory.sortToUtility();
	}

	private BidDetails getCounterOffer(double time) {
		List<BidDetails> myTop10Bids = this.myBidHistory.getNBestBids(10);
		List<BidDetails> opponentTop10Bids = this.opponentBidHistory
				.getNBestBids(10);
		double[][] comparisonMatrix = new double[10][10];

		for (int row = 0; row < opponentTop10Bids.size(); row++) {
			BidDetails opponentBidDetails = opponentTop10Bids.get(row);
			for (int col = 0; col < myTop10Bids.size(); col++) {
				BidDetails myBidDetails = myTop10Bids.get(col);
				// Compute Euclidean Distance between opponentBidDetails and
				// myBidDetails:
				try {
					comparisonMatrix[row][col] = computeEuclideanDistancePerBid(
							opponentBidDetails, myBidDetails, time);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		// Now traverse the comparisonMatrix, pick top 10 (least distanced) bids
		// per row (preserving some diversity) and add to a BidHistory HashMap.
		HashMap<Integer, BidDetails> hashMap = new HashMap<Integer, BidDetails>();
		for (int row = 0; row < 10; row++) {
			double closestBid = 99999.999;// some unrealistic number to
											// initialize.
			for (int col = 0; col < 10; col++) {
				if (col > 0) {
					if ((comparisonMatrix[row][col] < comparisonMatrix[row][col
							- 1])
							&& (comparisonMatrix[row][col] < closestBid)) {
						closestBid = comparisonMatrix[row][col];
						if (hashMap.containsKey(row)) {
							hashMap.remove(row);
							hashMap.put(row, myTop10Bids.get(col));
						}
					}
				} else {
					closestBid = comparisonMatrix[row][col];
					hashMap.put(row, myTop10Bids.get(col));
				}
			}
		}
		// Now find the bid with max utility for us
		double startUtiliy = 0.0;
		BidDetails selectedCounterOffer = null;
		Iterator iter = hashMap.keySet().iterator();
		while (iter.hasNext()) {
			Integer key = (Integer) iter.next();
			BidDetails bidDetails = hashMap.get(key);
			double bidUtility = bidDetails.getMyUndiscountedUtil();
			if (bidUtility > startUtiliy) {
				startUtiliy = bidUtility;
				selectedCounterOffer = bidDetails;
			}
		}
		return selectedCounterOffer;
	}

	private double computeEuclideanDistancePerBid(BidDetails opponentBidDetails,
			BidDetails myBidDetails, double time) throws Exception {
		Bid opponentBid = opponentBidDetails.getBid();
		Bid myBid = myBidDetails.getBid();
		double sum = 0.0;
		double opponentBidUtility;
		double myBidUtility;
		opponentBidUtility = this.myUtilitySpace
				.getUtilityWithDiscount(opponentBid, time);
		myBidUtility = this.myUtilitySpace.getUtilityWithDiscount(myBid, time);
		sum = Math.pow((opponentBidUtility - myBidUtility), 2.0);
		return Math.sqrt(sum);
	}

	private double getMyReservationValue(double time) {
		if (this.myUtilitySpace.getReservationValue() != null
				&& this.myUtilitySpace
						.getReservationValueWithDiscount(time) > 0.0d) {
			return this.myUtilitySpace.getReservationValueWithDiscount(time);
		} else {
			return this.defaultReservationValue;
		}
	}

	private Offer genearteCounterOffer(Offer opponentOffer, double time) {
		Offer counterOffer = null;
		if (time > 0.0d) {
			// //Sets our counterOffer in this.myLastBid. Resort to last Bid
			// until the following mechanism finds a different/conceding offer:
			if (this.myUtilitySpace.getUtilityWithDiscount(this.myLastBid,
					time) > this.myUtilitySpace.getUtilityWithDiscount(
							this.myBidHistory.getBestBidDetails().getBid(),
							time)) {
				counterOffer = new Offer(getAgentID(), this.myLastBid);
			} else {
				BidHistory tempBidHistory = this.myBidHistory
						.filterBetweenUtility(
								this.myUtilitySpace.getUtilityWithDiscount(
										this.myLastBid, time),
								this.myUtilitySpace.getUtilityWithDiscount(
										this.myBidHistory.getBestBidDetails()
												.getBid(),
										time));
				if (tempBidHistory != null && tempBidHistory.size() > 0) {
					tempBidHistory.sortToUtility();

					if (tempBidHistory.getMedianUtilityBid() != null) {
						this.myLastBid = tempBidHistory.getMedianUtilityBid()
								.getBid();
						counterOffer = new Offer(getAgentID(), this.myLastBid);

						if (!this.myUtilitySpace.isDiscounted()) {
							// Starting second level selection only for
							// non-discounted domains
							BidHistory anotherTempBidHistory = tempBidHistory
									.filterBetweenUtility(
											this.myUtilitySpace
													.getUtilityWithDiscount(
															tempBidHistory
																	.getMedianUtilityBid()
																	.getBid(),
															time),
											this.myUtilitySpace
													.getUtilityWithDiscount(
															tempBidHistory
																	.getBestBidDetails()
																	.getBid(),
															time));
							if (anotherTempBidHistory != null
									&& anotherTempBidHistory.size() > 0) {
								if (anotherTempBidHistory
										.getMedianUtilityBid() != null) {
									this.myLastBid = anotherTempBidHistory
											.getMedianUtilityBid().getBid();
									counterOffer = new Offer(getAgentID(),
											this.myLastBid);
								}
							}
							// Ended second level selection
						}
					}
				} else {
					counterOffer = new Offer(getAgentID(), this.myLastBid);// retain
																			// last
					// offer.
				}
			}

			try {
				// finds a different/conceding offer:
				BigDecimal concessionProbability;
				concessionProbability = Pconcede(
						this.myUtilitySpace.getUtilityWithDiscount(
								opponentOffer.getBid(), time),
						this.timeline.getTime());

				if ((this.hardlineCounter < this.alpha)
						&& (concessionProbability
								.compareTo(new BigDecimal(this.beta)) > 0)) {
					BidDetails tempBidDetails = null;
					tempBidDetails = getCounterOffer(time);
					Bid tempBid = tempBidDetails.getBid();
					double bidUtility = this.myUtilitySpace
							.getUtilityWithDiscount(tempBid, time);
					if (bidUtility > this.myUtilitySpace
							.getUtilityWithDiscount(this.bestBid, time)
							&& bidUtility >= this.getMyReservationValue(time)) {
						counterOffer = new Offer(getAgentID(), tempBid);
						this.myLastBid = tempBid;
					} else if (this.myUtilitySpace.getUtilityWithDiscount(
							this.bestBid,
							time) >= this.getMyReservationValue(time)) {
						counterOffer = new Offer(getAgentID(), this.bestBid);
						this.myLastBid = this.bestBid;
					}
				} else {
					if (this.counter > 1) // Not the first round.
					{
						if (this.myUtilitySpace.getUtilityWithDiscount(
								this.bestBid, time) < this.myUtilitySpace
										.getUtilityWithDiscount(this.myLastBid,
												time)) {
							counterOffer = new Offer(getAgentID(),
									this.myLastBid);
						} else if (this.myUtilitySpace.getUtilityWithDiscount(
								this.bestBid,
								time) >= this.getMyReservationValue(time)) {
							counterOffer = new Offer(getAgentID(),
									this.bestBid);
							this.myLastBid = this.bestBid;
						}
					} else {
						counterOffer = new Offer(getAgentID(), this.myLastBid);// will
																				// execute
						// just once
						// i.e., for
						// round 1.
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			try {
				counterOffer = new Offer(getAgentID(), this.bestBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return counterOffer;
	}

	private boolean acceptOpponentOffer(Offer opponentOffer, double time) {
		boolean result = false;
		double opponentOfferUtility;
		double myOfferUtility;
		double bestBidUtility;
		if (opponentOffer != null) {
			try {
				opponentOfferUtility = this.myUtilitySpace
						.getUtilityWithDiscount(opponentOffer.getBid(), time);
				myOfferUtility = this.myUtilitySpace
						.getUtilityWithDiscount(this.myLastBid, time);
				bestBidUtility = this.myUtilitySpace
						.getUtilityWithDiscount(this.bestBid, time);

				if (opponentOfferUtility >= this.getMyReservationValue(time)
						&& (2.1d * this.meanResponseTimeOfOpponent
								+ time >= 1.0d)) {
					if ((this.meanResponseTimeOfOpponent + time <= 1.0d)) {
						this.myLastBid = this.bestBid; // most probably our last
														// or last few counter
														// offer(s)!
						result = false;
					} else {
						result = true;
					}
				} else if (opponentOfferUtility >= 0.90d) {
					result = true;
				} else if ((opponentOfferUtility >= myOfferUtility)
						&& (opponentOfferUtility >= 0.85d) && time > 0.95d) {
					result = true;
				}

			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return result;
	}

	@Override
	public String getVersion() {
		return "1.1";
	}

	@Override
	public String getName() {
		return "AgentQuest";
	}

	/**
	 * This function determines the accept probability for an offer. At t=0 it
	 * will prefer high-utility offers. As t gets closer to 1, it will accept
	 * lower utility offers with increasing probability. it will never accept
	 * offers with utility 0.
	 * 
	 * @param u
	 *            is the utility
	 * @param t
	 *            is the time as fraction of the total available time (t=0 at
	 *            start, and t=1 at end time)
	 * @return the probability of an accept at time t
	 * @throws Exception
	 *             if you use wrong values for u or t.
	 * 
	 *             Function reused from Genius codebase.
	 */
	public BigDecimal Pconcede(double u, double t1) throws Exception {
		double t = t1 * t1 * t1; // steeper increase when deadline approaches.
		if (u < 0 || u > 1.05)
			throw new Exception("utility " + u + " outside [0,1]");
		// normalization may be slightly off, therefore we have a broad boundary
		// up to 1.05
		if (t < 0 || t > 1)
			throw new Exception("time " + t + " outside [0,1]");
		if (u > 1.)
			u = 1;
		if (t == 0.5) {
			if (!Double.isNaN(u)) {
				return new BigDecimal(u);
			} else
				throw new NumberFormatException(u + "");
		}
		double value = (u - 2. * u * t
				+ 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u * (-1. + 2 * t))))
				/ (-1. + 2 * t);

		if (!Double.isNaN(value)) {
			return new BigDecimal(value);
		} else
			throw new NumberFormatException(value + "");
	}

	public double sq(double x) {
		return x * x;
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}

}