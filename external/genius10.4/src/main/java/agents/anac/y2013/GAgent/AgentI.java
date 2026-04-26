package agents.anac.y2013.GAgent;

import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;

public class AgentI extends Agent {

	private Action actionOfPartner = null;
	private Range range = new Range(0D, 1D);
	private double time = 0;
	private BidHistory myBidHistory = new BidHistory();
	private SortedOutcomeSpace outcomeSpace;
	private BidHistory opponetBidHistory = new BidHistory();
	private static double MINIMUM_BID_UTILITY = 1D;
	private Random randomnr = new Random();
	private boolean isDiscount = false;
	private boolean isResavationValue = false;
	private double widthUtil = 0D;
	private boolean isFirst = true;
	private Probability prob;
	private double preOppponetUtil;
	private double var;
	private static double A = 0.9;

	private double sigmoidGain = 9;
	private double lowerLimit = 0.6;
	private double startTime = 0.7;
	private double tim = 0.0;

	@Override
	public void init() {

		if (utilitySpace.getDiscountFactor() != 0D) {
			isDiscount = true;
		}

		if (utilitySpace.getReservationValue() != 0D) {
			isResavationValue = true;
		}

		sigmoidGain = 2D;
		lowerLimit = 0.9;

		MINIMUM_BID_UTILITY = 1D;
		range.setLowerbound(changeThreshold(0));
		time = timeline.getTime();
		outcomeSpace = new SortedOutcomeSpace(utilitySpace);
	}

	public double changeThreshold(double time) {
		return (1 - lowerLimit + tim) / (1 + Math.exp(sigmoidGain * time))
				+ lowerLimit;
	}

	@Override
	public String getVersion() {
		return "3.1";
	}

	@Override
	public String getName() {
		return "GAgent";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;

		try {
			if (actionOfPartner == null) {
				action = new Offer(getAgentID(),
						utilitySpace.getMaxUtilityBid());
			}

			if (actionOfPartner instanceof Offer) {

				Bid partnerBid = ((Offer) actionOfPartner).getBid();
				BidDetails opponetBid = new BidDetails(partnerBid, 1);
				opponetBidHistory.add(opponetBid);

				// get my utility form opponet's bid
				double offeredUtilFromOpponent = getUtility(partnerBid);
				double offerundiscoutFromOpponent = utilitySpace
						.getUtility(partnerBid);

				if (isFirst) {
					widthUtil = 1 - offeredUtilFromOpponent;
					prob = new Probability(widthUtil);

					lowerLimit = 1 - widthUtil / 3;
					isFirst = false;
				} else {
					double diff = offerundiscoutFromOpponent - preOppponetUtil;
					var = prob.getVar(diff);
				}

				A = changeThreshold(time);

				preOppponetUtil = offerundiscoutFromOpponent;

				// get current time
				time = timeline.getTime();

				double a = 1 - var * widthUtil;

				if (a > A) {
					MINIMUM_BID_UTILITY = A;
				} else {
					if (var * widthUtil > (widthUtil / 2)) {
						MINIMUM_BID_UTILITY = 1 - widthUtil / 2;
					} else {
						MINIMUM_BID_UTILITY = a;
					}
				}

				range.setLowerbound(MINIMUM_BID_UTILITY);

				action = chooseRandomBidAction(time);

				Bid myBid = ((Offer) action).getBid();
				double myOfferedUtil = getUtility(myBid);

				if (isAcceptable(offerundiscoutFromOpponent, myOfferedUtil,
						time, partnerBid)) {
					action = new Accept(getAgentID(), partnerBid);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) action).getBid());
		}
		return action;
	}

	private boolean isAcceptable(double offeredUtilFromOpponent,
			double myOfferedUtil, double time, Bid parBid) throws Exception {

		BidDetails bd = new BidDetails(parBid, 1);

		if (myBidHistory.getHistory().contains(bd)) {
			return true;
		}

		if (offeredUtilFromOpponent > MINIMUM_BID_UTILITY) {
			return true;
		}

		if (isResavationValue) {
			if (getUtility(parBid) < lowerLimit) {
				return false;
			}
		}

		if (time > 0.9988) {
			Bid nextBid;
			List<BidDetails> bdd = opponetBidHistory.getNBestBids(5);
			nextBid = bdd.get(4).getBid();
			double utility = getUtility(nextBid);

			if (offeredUtilFromOpponent > utility) {
				return true;
			}
		}

		// double P = Paccept(offeredUtilFromOpponent,time);

		// if (P > 0.95)
		// return true;

		return false;

	}

	private Action chooseRandomBidAction(double time) {
		Bid nextBid = null;
		try {
			if (time > 0.9988) {
				List<BidDetails> bdd = opponetBidHistory.getNBestBids(3);
				int opt = randomnr.nextInt(3);
				nextBid = bdd.get(opt).getBid();

				if (isResavationValue) {
					if (getUtility(nextBid) < lowerLimit) {
						nextBid = getRandomBid();
					}
				}

			} else {
				nextBid = getRandomBid();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (nextBid == null)
			return new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		return (new Offer(getAgentID(), nextBid));
	}

	private Bid getRandomBid() throws Exception {
		List<BidDetails> rangeBid = outcomeSpace.getBidsinRange(range);
		int rangeBidSize = rangeBid.size();
		Bid offerbid;
		BidDetails offerBidDetail;

		if (rangeBidSize == 1) {
			offerbid = rangeBid.get(0).getBid();
			offerBidDetail = new BidDetails(offerbid, 1);
		} else {
			int bitCount = 0;
			offerBidDetail = rangeBid.get(bitCount);
			while (myBidHistory.getHistory().contains(offerBidDetail)) {
				int ran = randomnr.nextInt(rangeBidSize - 1);
				offerBidDetail = rangeBid.get(ran);
				bitCount++;
				if (bitCount > 60) {
					break;
				}

			}
		}
		myBidHistory.add(offerBidDetail);
		offerbid = offerBidDetail.getBid();
		return offerbid;

	}

	double Paccept(double u, double t1) throws Exception {
		double t = t1 * t1 * t1; // steeper increase when deadline approaches.
		if (u < 0 || u > 1.05)
			throw new Exception("utility " + u + " outside [0,1]");
		// normalization may be slightly off, therefore we have a broad boundary
		// up to 1.05
		if (t < 0 || t > 1)
			throw new Exception("time " + t + " outside [0,1]");
		if (u > 1.)
			u = 1;
		if (t == 0.5)
			return u;
		return (u - 2. * u * t
				+ 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u * (-1. + 2 * t))))
				/ (-1. + 2 * t);
	}

	double sq(double x) {
		return x * x;
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2013";
	}
}
