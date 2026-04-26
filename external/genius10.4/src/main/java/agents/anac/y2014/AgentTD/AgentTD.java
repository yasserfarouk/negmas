package agents.anac.y2014.AgentTD;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
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
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.timeline.Timeline;

public class AgentTD extends Agent {
	private Action actionOfPartner = null;
	private static double MINIMUM_BID_UTILITY = 0.0;
	private Bid maximumBidFromPartner = null;
	private Action temp_action = null;
	private Bid temp_bid = null;
	private static double DISCOUNT_FACTOR = 0.0;
	private static boolean first = true;

	@Override
	public void init() {
		MINIMUM_BID_UTILITY = utilitySpace.getReservationValueUndiscounted();
		DISCOUNT_FACTOR = utilitySpace.getDiscountFactor();
	}

	@Override
	public String getVersion() {
		return "3.1";
	}

	@Override
	public String getName() {
		return "AgentTD";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			if (actionOfPartner == null || first) {
				temp_action = chooseFirstBidAction();
				action = temp_action;
				first = false;
			}
			if (actionOfPartner instanceof Offer) {
				Bid partnerBid = ((Offer) actionOfPartner).getBid();
				double offeredUtilFromOpponent = getUtility(partnerBid);
				if (offeredUtilFromOpponent > MINIMUM_BID_UTILITY) {
					MINIMUM_BID_UTILITY = offeredUtilFromOpponent;
				}

				if (maximumBidFromPartner == null) {
					maximumBidFromPartner = partnerBid;
				} else {
					if (getUtility(maximumBidFromPartner) < getUtility(
							partnerBid))
						maximumBidFromPartner = partnerBid;
				}

				double time = timeline.getTime();
				if (time < 0.95) {
					action = chooseRandomBidAction();
				} else {
					action = new Offer(getAgentID(), maximumBidFromPartner);
				}

				if (isAcceptable(offeredUtilFromOpponent, time)) {
					action = new Accept(getAgentID(), partnerBid);
				}
				if (isEndNegotiation(time)) {
					action = new EndNegotiation(getAgentID());
				}

			}

			if (timeline.getType().equals(Timeline.Type.Time)) {
				sleep(0.005);
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}

		return action;
	}

	private boolean isAcceptable(double offeredUtilFromOpponent, double time)
			throws Exception {

		if (offeredUtilFromOpponent > getUtility(maximumBidFromPartner)) {
			if (time < 0.7) {
				if (offeredUtilFromOpponent > 0.85)
					return true;
			} else if (time < 0.98) {
				if (offeredUtilFromOpponent > 0.75)
					return true;
			} else {
				return true;
			}
		}
		return false;
	}

	private boolean isEndNegotiation(double time) throws Exception {
		if (utilitySpace.getUtilityWithDiscount(maximumBidFromPartner,
				time) < utilitySpace.getReservationValueWithDiscount(time))
			return true;
		return false;
	}

	private Action chooseRandomBidAction() {
		Bid nextBid = null;
		try {
			nextBid = getRandomBid();
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null)
			return (new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid()));
		return (new Offer(getAgentID(), nextBid));
	}

	private Action chooseFirstBidAction() {
		Bid nextBid = null;
		try {
			nextBid = getFirstBid();
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null) {
			return (new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid()));
		} else {
			temp_bid = nextBid;
		}
		return (new Offer(getAgentID(), nextBid));
	}

	private Bid getFirstBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();
		Bid bid = null;
		Bid max_bid = null;
		double max_bid_utility = 0;
		int count = 0;

		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr
							.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(
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
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(),
							new ValueInteger(optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported ");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
			if (getUtility(bid) >= max_bid_utility) {
				max_bid_utility = getUtility(bid);
				max_bid = bid;
			}

			count++;
		} while (count < 1000000);
		return max_bid;
	}

	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		Bid bid = null;
		double temp = 0.0;
		int count = 0;

		if (MINIMUM_BID_UTILITY > 0.5) {
			temp = MINIMUM_BID_UTILITY;
		} else {
			temp = getUtility(temp_bid) - 0.1;
		}

		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr
							.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(
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
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(),
							new ValueInteger(optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported ");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (getUtility(bid) < temp);
		return bid;
	}

	double sq(double x) {
		return x * x;
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}
