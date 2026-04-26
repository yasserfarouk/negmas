package agents.anac.y2012.MetaAgent.agents.LYY;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * @author Yaron, Yulia, Lena Some modifications to the standard Simple Agent.
 */
public class LYYAgent extends Agent {
	Random randomGenerator = new Random(System.nanoTime());

	private Action actionOfPartner = null;

	private static double FIRST_DECISION_LEVEL = 0.7; // Utilities below this
														// threshold are
														// rejected.
	private static double SECOND_DECISION_LEVEL = 0.8; // Two levels of random
														// decision.
	private static double ACCEPT_OFFER_LEVEL = 0.85; // Utilities above this
														// threshold are
														// accepted.

	@Override
	public String getVersion() {
		return "1.0";
	}

	public String getName() {
		return "LYYAgent";
	}

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	public void init() {
		if (utilitySpace.getReservationValue() != null)
			FIRST_DECISION_LEVEL = utilitySpace.getReservationValue();
	}

	public void ReceiveMessage(Action partnerAction) {
		/*
		 * System.out.println("LYY Agent: ReceiveMessage - Start");
		 * System.out.flush();
		 * 
		 * if (partnerAction instanceof Offer)
		 * System.out.println("LYY Agent: Got Offer"); else if (partnerAction
		 * instanceof Accept) System.out.println("LYY Agent: Got Accept"); else
		 * System.out.println("LYY Agent: End Negotiation???");
		 * 
		 * System.out.println("LYY Agent: ReceiveMessage - End");
		 * System.out.flush();
		 */

		actionOfPartner = partnerAction;
	}

	public Action chooseAction() {
		// System.out.println("LYY Agent: chooseAction - Start");
		// System.out.flush();
		Action action = null;
		try {
			if (actionOfPartner == null) {
				action = chooseRandomBidAction();
				// System.out.println("LYY Agent: I'm the negotiation's beginner");
			} else if (actionOfPartner instanceof Offer) {
				// Get the partner's bid.
				Bid partnerBid = ((Offer) actionOfPartner).getBid();
				double offeredUtililty = getUtility(partnerBid);
				double time = timeline.getTime();

				// System.out.println("LYY Agent: Consider the partner's bid - Bid="+partnerBid);

				// Consider acceptance of the partner's bid.
				if (isAcceptable(offeredUtililty, time))
					action = new Accept(getAgentID(), partnerBid);

				// If the partner offered an unacceptable offer, return a random
				// bid.
				else
					action = chooseRandomBidAction();
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		// System.out.println("LYY Agent: chooseAction - End");
		// System.out.flush();
		return action;
	}

	private boolean isAcceptable(double offeredUtililty, double time)
			throws Exception {
		double squredTime = time * time;

		// System.out.println("LYY Agent: isAcceptable (util="+offeredUtililty+", time="+time+", time^2="+squredTime+")");
		// System.out.flush();

		// Test values validity.
		if ((offeredUtililty < 0) || (offeredUtililty > 1.05))
			throw new Exception("utility " + offeredUtililty + " outside [0,1]");
		if ((time <= 0) || (time > 1))
			throw new Exception("time " + time + " outside [0,1]");

		// Reject low utilities.
		if (offeredUtililty < FIRST_DECISION_LEVEL)
			return false;

		// Accept high utilities.
		if (offeredUtililty >= ACCEPT_OFFER_LEVEL)
			return true;

		// No early compromise.
		if (time < 0.005)
			return false;

		// Get random factor.
		double rand = randomGenerator.nextDouble();

		// System.out.println("LYY Agent: random factor="+rand);
		// System.out.flush();

		if (offeredUtililty < SECOND_DECISION_LEVEL)
			return (rand < squredTime);

		// System.out.println("LYY Agent: sqrt(time)="+Math.sqrt(time));
		// System.out.flush();

		return (rand < Math.sqrt(time));
	}

	/**
	 * Wrapper for getRandomBid, for convenience.
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
			return (new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid()));
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

		double utilDiff = ACCEPT_OFFER_LEVEL - FIRST_DECISION_LEVEL;
		double currTime = timeline.getTime();
		double decreaseFactor = currTime * currTime;
		double utilFloor = ACCEPT_OFFER_LEVEL - decreaseFactor * utilDiff;

		/*
		 * System.out.println("LYY Agent: getRandomBid - Start(time="+currTime+
		 * ", diff="+utilDiff+", floor="+utilFloor+")"); System.out.flush();
		 */

		// createFrom a random bid with utility>SECOND_DECISION_LEVEL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (2 minutes)
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
							+ " not supported by LYYAgent");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (getUtility(bid) < utilFloor);

		/*
		 * System.out.println("LYY Agent: getRandomBid - End (my bid util="+
		 * getUtility(bid)+")"); System.out.flush();
		 */
		return bid;
	}
}
