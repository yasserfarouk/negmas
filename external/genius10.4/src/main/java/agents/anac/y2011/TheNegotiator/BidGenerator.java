package agents.anac.y2011.TheNegotiator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;

/**
 * The BidsGenerator class is used to generate a offer.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class BidGenerator {

	// reference to the bidscollection
	private BidsCollection bidsCollection;
	// reference to the negotiation agent
	private Agent agent;
	// a factor which determines when to ignore the upperbound for threshold
	private double randomMove = 0.3f;
	private Random random100;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Creates a BidGenerator-object which determines which offer should be
	 * made.
	 * 
	 * @param partner
	 *            agent
	 * @param bidsCollection
	 *            of all possible bids and the partner bids
	 */
	public BidGenerator(Agent agent, BidsCollection bidsCollection) {
		this.agent = agent;
		this.bidsCollection = bidsCollection;
		createAllBids();
		this.bidsCollection.sortPossibleBids();
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
		} else {
			random100 = new Random();
		}
	}

	/**
	 * Determine what (counter)offer should be made in a given phase with a
	 * minimum threshold.
	 * 
	 * @param agentID
	 *            of our agent
	 * @param phase
	 *            of the negotation
	 * @param minimum
	 *            threshold
	 * @return (counter)offer
	 */
	public Action determineOffer(AgentID agentID, int phase, double threshold) {
		Bid bid = null;
		double upperThreshold = bidsCollection.getUpperThreshold(threshold,
				0.20);
		if (phase == 1) {
			// if the random value is above the random factor, do a normal move
			if (random100.nextDouble() > randomMove) {
				// do a normal move, which is a move between an threshold
				// interval
				bid = bidsCollection
						.getOwnBidBetween(threshold, upperThreshold);
			} else {
				// do a move which can be oppertunistic (ignore upperbound)
				bid = bidsCollection.getOwnBidBetween(upperThreshold - 0.00001,
						1.1);
			}
		} else { // phase 2 or 3

			// play best moves of opponent if above threshold
			bid = bidsCollection.getBestPartnerBids(threshold);
			// could be that there is no opponent bid above the threshold
			if (bid == null) {
				if (random100.nextDouble() > randomMove) {
					bid = bidsCollection.getOwnBidBetween(threshold,
							upperThreshold);
				} else {
					bid = bidsCollection.getOwnBidBetween(
							upperThreshold - 0.00001, 1.1);
				}
			}
		}
		Action action = new Offer(agentID, bid);
		return action;
	}

	/**
	 * Create all possible bids using a call to the recursive Cartestian product
	 * options generator.
	 */
	private void createAllBids() {
		List<Issue> issues = agent.utilitySpace.getDomain().getIssues();

		ArrayList<IssueDiscrete> discreteIssues = new ArrayList<IssueDiscrete>();

		for (Issue issue : issues) {
			discreteIssues.add((IssueDiscrete) issue);
		}

		ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> result = generateAllBids(
				discreteIssues, 0);

		for (ArrayList<Pair<Integer, ValueDiscrete>> bidSet : result) {
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			for (Pair<Integer, ValueDiscrete> pair : bidSet) {
				values.put(pair.getFirst(), pair.getSecond());
			}
			try {
				Bid bid = new Bid(agent.utilitySpace.getDomain(), values);
				double utility = agent.utilitySpace.getUtility(bid);
				bidsCollection.addPossibleBid(bid, utility);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * The recursive Cartestian product options generator. Generates all
	 * possible bids.
	 * 
	 * @param issueList
	 * @param i
	 *            , parameter used in the recursion
	 * @return a list of a list with pairs of integer (issue at stake) and a
	 *         value (the chosen option)
	 */
	private ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> generateAllBids(
			ArrayList<IssueDiscrete> issueList, int i) {

		// stop condition
		if (i == issueList.size()) {
			// return a list with an empty list
			ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> result = new ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>>();
			result.add(new ArrayList<Pair<Integer, ValueDiscrete>>());
			return result;
		}

		ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> result = new ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>>();
		ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> recursive = generateAllBids(
				issueList, i + 1); // recursive call

		// for each element of the first list of input
		for (int j = 0; j < issueList.get(i).getValues().size(); j++) {
			// add the element to all combinations obtained for the rest of the
			// lists
			for (int k = 0; k < recursive.size(); k++) {
				// copy a combination from recursive
				ArrayList<Pair<Integer, ValueDiscrete>> newList = new ArrayList<Pair<Integer, ValueDiscrete>>();
				for (Pair<Integer, ValueDiscrete> set : recursive.get(k)) {
					newList.add(set);
				}
				// add element of the first list
				ValueDiscrete value = issueList.get(i).getValues().get(j);
				int issueNr = issueList.get(i).getNumber();
				newList.add(new Pair<Integer, ValueDiscrete>(issueNr, value));

				// add new combination to result
				result.add(newList);
			}
		}
		return result;
	}
}