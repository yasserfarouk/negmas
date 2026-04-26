package genius.core.boaframework;

import java.util.ArrayList;
import java.util.Map;

import genius.core.Bid;
import genius.core.analysis.BidPointTime;

/**
 * The MAC is a tool which allows to test many acceptance strategies in the same
 * negotiation trace. Each AC generates an outcome, which is saved separately.
 * Note that while this tool allows to test a large amount of AC's in the same
 * trace, there is a computational cost. Therefore we recommend to use at most
 * 50 AC's.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class Multi_AcceptanceCondition extends AcceptanceStrategy {

	// list of acceptance strategies to checked for acceptance
	protected ArrayList<AcceptanceStrategy> ACList;
	// list of outcomes; an outcome is added when an AC accepts, or the
	// negotiation ended
	protected ArrayList<OutcomeTuple> outcomes;

	private boolean firstTime = false;
	private boolean startingAgent = false;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public Multi_AcceptanceCondition() {
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		// this.negotiationSession = negoSession;
		outcomes = new ArrayList<OutcomeTuple>();
		ACList = new ArrayList<AcceptanceStrategy>();

		/*
		 * // EXAMPLE: add 36 variants of the AC_CombiV4 acceptance condition
		 * for(int d = 0; d <= 35; d += 1) { ACList.add(new
		 * AC_CombiV4(negoSession, strat, 1, 0, 0.85 + d * 0.01, 0, 0.95)); }
		 */
	}

	/**
	 * The main method of the MAC. This method ensures that every acceptance
	 * condition is checked, and that the outcome for each AC is saved.
	 */
	@Override
	public Actions determineAcceptability() {

		if (!firstTime && (negotiationSession.getOpponentBidHistory().size() == negotiationSession.getOwnBidHistory()
				.size())) {
			startingAgent = true;
		}

		ArrayList<AcceptanceStrategy> acceptors = new ArrayList<AcceptanceStrategy>();
		for (AcceptanceStrategy a : ACList) {
			Bid lastOpponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid();
			String name = a.getClass().getSimpleName() + " " + printParameters(a);
			double time = negotiationSession.getTime();
			OutcomeTuple outcome;
			if (a.determineAcceptability().equals(Actions.Accept)) {
				System.out.println(name + " accepted Bid" + " Util: "
						+ negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil());

				// sets the amount of bids made for agent A and agent B
				if (startingAgent) {
					outcome = new OutcomeTuple(lastOpponentBid, name, time,
							negotiationSession.getOwnBidHistory().size(),
							negotiationSession.getOpponentBidHistory().size(), "accept", "agentA");
				} else {
					outcome = new OutcomeTuple(lastOpponentBid, name, time,
							negotiationSession.getOpponentBidHistory().size(),
							negotiationSession.getOwnBidHistory().size(), "accept", "agentB");

				}
				outcomes.add(outcome);
				acceptors.add(a);
			} else if (a.determineAcceptability().equals(Actions.Break)) {
				outcome = new OutcomeTuple(lastOpponentBid, name, time, negotiationSession.getOwnBidHistory().size(),
						negotiationSession.getOpponentBidHistory().size(), "breakoff", "");
				outcomes.add(outcome);
				acceptors.add(a);
			}

		}
		ACList.removeAll(acceptors);

		if (ACList.isEmpty()) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	public ArrayList<OutcomeTuple> getOutcomes() {
		return outcomes;
	}

	public ArrayList<AcceptanceStrategy> getACList() {
		return ACList;
	}

	public String printParameters(AcceptanceStrategy a) {
		return a.printParameters();
	}

	@Override
	public String printParameters() {
		return null;
	}

	@Override
	public boolean isMAC() {
		return true;
	}

	public void remainingACAccept(Bid lastBid, double time, ArrayList<BidPointTime> fAgentABids,
			ArrayList<BidPointTime> fAgentBBids, String acceptedBy) {
		ArrayList<AcceptanceStrategy> acceptors = new ArrayList<AcceptanceStrategy>();

		for (AcceptanceStrategy strat : ACList) {
			String name = strat.getClass().getSimpleName() + " " + printParameters(strat);
			OutcomeTuple tuple = new OutcomeTuple(lastBid, name, time, fAgentABids.size(), fAgentBBids.size(), "accept",
					acceptedBy);
			outcomes.add(tuple);
			acceptors.add(strat);

		}
		ACList.removeAll(acceptors);
	}

	public void remainingAC(String type) {
		ArrayList<AcceptanceStrategy> acceptors = new ArrayList<AcceptanceStrategy>();

		for (AcceptanceStrategy strat : ACList) {
			String name = strat.getClass().getSimpleName() + " " + printParameters(strat);
			OutcomeTuple tuple = new OutcomeTuple(null, name, 1, -1, -1, type, "");
			outcomes.add(tuple);
			acceptors.add(strat);

		}
		ACList.removeAll(acceptors);
	}

	@Override
	public String getName() {
		return "Multi AcceptanceCondition";
	}
}