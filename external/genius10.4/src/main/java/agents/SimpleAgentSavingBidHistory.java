package agents;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * @author S. Hourmann Some improvements over the standard Agent. Saving Bid
 *         History for the session.
 * 
 *         Random Walker, Zero Intelligence Agent
 */
public class SimpleAgentSavingBidHistory extends Agent {
	// "state" represents "where am I in the code".
	// I want to print "state" when I print a message about saving data.
	private String state;
	private Action actionOfPartner = null;

	/**
	 * Note: {@link SimpleAgentSavingBidHistory} does not account for the
	 * discount factor in its computations
	 */
	private static double MINIMUM_BID_UTILITY = 0.0;

	private BidHistory currSessOppBidHistory;
	private BidHistory prevSessOppBidHistory;

	public SimpleAgentSavingBidHistory() {
		super();
		this.currSessOppBidHistory = new BidHistory();
	}

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	public void init() {
		MINIMUM_BID_UTILITY = utilitySpace.getReservationValueUndiscounted();
		myBeginSession();
	}

	public void myBeginSession() {
		System.out.println("Starting match num: " + sessionNr);

		// ---- Code for trying save and load functionality
		// First try to load saved data
		// ---- Loading from agent's function "loadSessionData"
		Serializable prev = this.loadSessionData();
		if (!(prev == null)) {
			prevSessOppBidHistory = (BidHistory) prev;
			System.out
					.println("---------/////////// NEW  NEW  NEW /////////////----------");
			System.out.println("The size of the previous BidHistory is: "
					+ prevSessOppBidHistory.size());
		} else {
			// If didn't succeed, it means there is no data for this preference
			// profile
			// in this domain.
			System.out.println("There is no history yet.");
		}
	}

	@Override
	public String getVersion() {
		return "3.1";
	}

	@Override
	public String getName() {
		return "Simple Agent Saving Bid History";
	}

	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
		if (opponentAction instanceof Offer) {
			Bid bid = ((Offer) opponentAction).getBid();
			// 2. store the opponent's trace
			try {
				BidDetails opponentBid = new BidDetails(bid,
						utilitySpace.getUtility(bid), timeline.getTime());
				currSessOppBidHistory.add(opponentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public Action chooseAction() {
		Action action = null;
		Bid partnerBid = null;
		try {
			if (actionOfPartner == null)
				action = chooseRandomBidAction();
			if (actionOfPartner instanceof Offer) {
				partnerBid = ((Offer) actionOfPartner).getBid();
				double offeredUtilFromOpponent = getUtility(partnerBid);
				// get current time
				double time = timeline.getTime();
				action = chooseRandomBidAction();

				Bid myBid = ((Offer) action).getBid();
				double myOfferedUtil = getUtility(myBid);

				// accept under certain circumstances
				if (isAcceptable(offeredUtilFromOpponent, myOfferedUtil, time)) {
					action = new Accept(getAgentID(), partnerBid);

					// ---- Code for trying save and load functionality
					// /////////////////////////////////
					state = "I accepted so I'm trying to save. ";
					tryToSaveAndPrintState();
					// /////////////////////////////////

				}
			}
			if (actionOfPartner instanceof EndNegotiation) {

				// ---- Code for trying save and load functionality
				// /////////////////////////////////
				state = "Got EndNegotiation from opponentttttttttt. ";
				tryToSaveAndPrintState();
				// /////////////////////////////////
			}
			sleep(0.005); // just for fun
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());

			// ---- Code for trying save and load functionality
			// /////////////////////////////////
			state = "Got Exceptionnnnnnnnnn. ";
			tryToSaveAndPrintState();
			// /////////////////////////////////
			// best guess if things go wrong.
			action = new Accept(getAgentID(), partnerBid);
		}
		return action;
	}

	// ---- Code for trying save and load functionality
	private void tryToSaveAndPrintState() {

		// ---- Saving from agent's function "saveSessionData"
		this.saveSessionData(currSessOppBidHistory);
		System.out.println(state + "The size of the BidHistory I'm saving is: "
				+ currSessOppBidHistory.size());
	}

	private boolean isAcceptable(double offeredUtilFromOpponent,
			double myOfferedUtil, double time) throws Exception {
		double P = Paccept(offeredUtilFromOpponent, time);
		if (P > Math.random())
			return true;
		return false;
	}

	/**
	 * Wrapper for getRandomBid, for convenience.
	 * 
	 * @return new Action(Bid(..)), with bid utility > MINIMUM_BID_UTIL. If a
	 *         problem occurs, it returns an Accept() action.
	 */
	private Action chooseRandomBidAction() {
		Bid nextBid = null;
		try {
			nextBid = getRandomBid();
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null)
			return (new Accept(getAgentID(), currSessOppBidHistory.getLastBid()));
		return (new Offer(getAgentID(), nextBid));
	}

	/**
	 * @return a random bid with high enough utility value.
	 * @throws Exception
	 *             if we can't compute the utility (eg no evaluators have been
	 *             set) or when other evaluators than a DiscreteEvaluator are
	 *             present in the util space.
	 */
	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		// createFrom a random bid with utility>MINIMUM_BID_UTIL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (3 minutes)
		// but this is just a simple agent.
		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete
							.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal
							.getNumberOfDiscretizationSteps() - 1);
					values.put(
							lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal
											.getLowerBound())
									* (double) (optionInd)
									/ (double) (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(
							optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by SamantaAgent2");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (getUtility(bid) < MINIMUM_BID_UTILITY);

		return bid;
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
	 */
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
		return (u - 2. * u * t + 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u
				* (-1. + 2 * t))))
				/ (-1. + 2 * t);
	}

	double sq(double x) {
		return x * x;
	}
}
