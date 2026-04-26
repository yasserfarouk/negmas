package agents.anac.y2013.SlavaAgent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * 
 * @author Slava Bronfman, Moshe Hazoom & Guy Dubrovski
 * 
 */
public class SlavaAgent extends Agent {
	// Last action of partner.
	private Action actionOfPartner = null;

	// Best bid offered by the opponent on this negotiation.
	private Bid maxBidRecieved = null;

	// Best bid for us.
	private Bid maxBid;

	// Stores all the best bids for us that their utility is more than
	// UTILITY_THRESHOLD
	private Map<Bid, Double> bestBidsMap;
	private List<Bid> bidsAsArray;

	// Best offer (the opponent offered). From disk, on previous negotiations.
	private Bid bestOffer;

	// Number of iterations in order to calculate the next random bid.
	private final int MAX_ITERATIONS = 10000;

	// Period of time we are doing exploration.
	private final double EXPLORATION_RATE = 0.95;

	// The threshold from which we offer bids to the opponent.
	private final double UTILITY_THRESHOLD = 0.95;

	// Don't accept if utility of the opponent is lower than these value.
	private final double MIN_UTILITY_ACCEPT = 0.7;

	// If the utility of the opponent is higher than these value, accept
	// immediately.
	private final double GOOD_ENOUGHT_UTILITY = 0.9;

	/**
	 * Method calculates the bid that has a maximum value for us from the
	 * domain.
	 * 
	 * @return
	 * @throws Exception
	 */
	public Bid GetMaxBid() throws Exception {

		Bid max = utilitySpace.getDomain().getRandomBid(null);
		Bid tempBidding = utilitySpace.getDomain().getRandomBid(null);

		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {

			double maxUtil = 0;
			int indexOfMaximumValue = 0;

			Issue currIssue = utilitySpace.getDomain().getIssues().get(i);

			if (currIssue.getType().equals(ISSUETYPE.INTEGER)) {
				IssueInteger issueInteger = (IssueInteger) currIssue;
				tempBidding = tempBidding.putValue(currIssue.getNumber(),
						new ValueInteger(issueInteger.getUpperBound()));
				maxUtil = utilitySpace.getUtility(tempBidding);
				tempBidding = tempBidding.putValue(currIssue.getNumber(),
						new ValueInteger(issueInteger.getLowerBound()));
				double minUtil = utilitySpace.getUtility(tempBidding);
				if (maxUtil > minUtil) {
					max = max.putValue(currIssue.getNumber(),
							new ValueInteger(issueInteger.getUpperBound()));
				} else {
					max = max.putValue(currIssue.getNumber(),
							new ValueInteger(issueInteger.getLowerBound()));
				}

			} else if (currIssue.getType().equals(ISSUETYPE.REAL)) {
				IssueReal issueReal = (IssueReal) currIssue;
				tempBidding = tempBidding.putValue(currIssue.getNumber(),
						new ValueReal(issueReal.getUpperBound()));
				maxUtil = utilitySpace.getUtility(tempBidding);
				tempBidding = tempBidding.putValue(currIssue.getNumber(),
						new ValueReal(issueReal.getLowerBound()));
				double minUtil = utilitySpace.getUtility(tempBidding);
				if (maxUtil > minUtil) {
					max = max.putValue(currIssue.getNumber(),
							new ValueReal(issueReal.getUpperBound()));
				} else {
					max = max.putValue(currIssue.getNumber(),
							new ValueReal(issueReal.getLowerBound()));
				}

			} else if (currIssue.getType().equals(ISSUETYPE.DISCRETE)) {
				IssueDiscrete issueDiscrete = (IssueDiscrete) currIssue;
				for (int j = 0; j < issueDiscrete.getNumberOfValues(); j++) {
					tempBidding = tempBidding.putValue(currIssue.getNumber(),
							issueDiscrete.getValue(j));
					double tempUtil = utilitySpace.getUtility(tempBidding);
					if (tempUtil > maxUtil) {
						indexOfMaximumValue = j;
						maxUtil = tempUtil;
					}
				}
				max = max.putValue(currIssue.getNumber(),
						issueDiscrete.getValue(indexOfMaximumValue));
			}
		}

		return (max);
	}

	/**
	 * Initialize our parameters for the agent:
	 * 
	 * maxBid - the bid that maximized our profit which we have found so far.
	 * bestOffer - the best bid for us that this opponent offered us from
	 * previous negotiations. bestBidsMap - stores all the best bids (those who
	 * has utility higher than UTILITY_THRESHOLD).
	 */
	@Override
	public void init() {
		try {

			// Initialize the best bid for use with a random bid.
			this.maxBid = this.utilitySpace.getDomain().getRandomBid(null);

			// Get the best bids for us that are more than a threshold.
			this.bestBidsMap = this.getBestBidsForUs();
			this.bidsAsArray = new ArrayList<Bid>(this.bestBidsMap.keySet());
		} catch (Exception e) {
			e.printStackTrace();
		}

		Serializable previousOffers = this.loadSessionData();

		// There exists a previous offer from before
		if (previousOffers != null) {
			this.bestOffer = (Bid) previousOffers;
		}
	}

	/**
	 * Version of the agent.
	 * 
	 * @return
	 */
	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "Slava Agent";
	}

	/**
	 * Method saves the opponent's action. If it's an offer, it calculates its
	 * utility. If the utility is higher than all previous bids we have seen so
	 * far from this opponent, we are receiveMessage the variables's value.
	 * 
	 * If it's the first time: If we played against this agent, we have a value
	 * on the variable "bestOffer" and we initialize the parameter
	 * "maxBidRevieved" to it (of course, only if its lower than it). Otherwise,
	 * we initialize "bestOffer" to be that bid.
	 */
	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;

		// We are the first to offer a bid.
		if (actionOfPartner == null) {
			return;
		}

		// The opponent offerers a bid.
		if (actionOfPartner instanceof Offer) {
			Offer opponentOffer = (Offer) actionOfPartner;
			Bid bid = opponentOffer.getBid();
			try {
				double utility = this.utilitySpace.getUtility(bid);

				// Save the best bid that the opponent offered to us.
				if (this.maxBidRecieved == null) { // The first offer.
					this.maxBidRecieved = bid;

					// Initialize the parameter we will later save to disk.
					if (this.bestOffer == null) {
						this.bestOffer = this.maxBidRecieved;
						// Initialize the maximum bid received from the opponent
						// (only if its lower than previous negotiations).
					} else if (this.utilitySpace
							.getUtility(this.maxBidRecieved) < this.utilitySpace
									.getUtility(this.bestOffer)) {
						this.maxBidRecieved = this.bestOffer;
					}
				} else if (utility > this.utilitySpace
						.getUtility(this.maxBidRecieved)) { // Not
															// the
															// first
															// offer.
					this.maxBidRecieved = bid;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * Choose the next action to make for agent.
	 * 
	 * If it's an offer that has utility more than GOOD_ENOUGHT_UTILITY, accept
	 * immediately.
	 * 
	 * Otherwise, split to 2: Exploration and exploitation. In exploration part,
	 * receiveMessage the maximal bid for us (if found) and with probability of
	 * 0.5, offer it to the opponent and with probability of 0.5, offer to him a
	 * bid that is good enough for us randomly (one that has utility more than
	 * UTILITY_THRESHOLD). In exploitation part, offer always the best bid for
	 * us. Accept if the opponent offers a bid that has utility more than
	 * MIN_UTILITY_ACCEPT and has value higher or equals to all his previous
	 * offers. Otherwise, offer the best bid for us.
	 * 
	 * On exceptions, offer the best bid for us.
	 */
	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			// We are the first to choose an action
			if (actionOfPartner == null) {
				this.maxBid = this.calculateNextBid();
				action = new Offer(getAgentID(), this.maxBid);
			}
			if (actionOfPartner instanceof Offer) {
				// If the user offered a bid that is good enough for us, accept.
				Offer opponentOffer = (Offer) actionOfPartner;
				Bid bid = opponentOffer.getBid();
				double utility = this.utilitySpace.getUtility(bid);
				if (utility >= this.GOOD_ENOUGHT_UTILITY) {
					action = new Accept(this.getAgentID(),
							((ActionWithBid) actionOfPartner).getBid());
					return (action);
				}

				// if the time is < EXPLORATION_RATE always offer a max bid
				// until so
				if (this.timeline.getTime() <= this.EXPLORATION_RATE) {
					this.maxBid = this.calculateNextBid();

					Random rand = new Random();

					// In a probability of 0.5, offer the best bid
					if (rand.nextDouble() <= 0.5) {
						action = new Offer(getAgentID(), this.maxBid);
						// In a probability of 0.5, offer one of the other best
						// bids.
					} else {
						Bid nextBid = this.bidsAsArray
								.get((rand.nextInt(this.bidsAsArray.size())));
						action = new Offer(getAgentID(), nextBid);
					}
				} else // Exploitation
				{
					opponentOffer = (Offer) actionOfPartner;
					bid = opponentOffer.getBid();
					try {
						utility = this.utilitySpace.getUtility(bid);

						// If the user offered a bid that is more than the
						// maximum so far, accept.
						if (utility >= this.MIN_UTILITY_ACCEPT
								&& utility >= this.utilitySpace
										.getUtility(this.maxBidRecieved)) {
							action = new Accept(this.getAgentID(),
									((ActionWithBid) actionOfPartner).getBid());
						} else { // Offerers the maximum bid for us.
							action = new Offer(getAgentID(), this.maxBid);
						}
					} catch (Exception e) {
						e.printStackTrace();
						action = new Offer(getAgentID(), this.maxBid);
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			this.maxBid = this.calculateNextBid();
			action = new Offer(getAgentID(), this.maxBid);
		}

		// Sleep a little bit in order to see the results.
		this.sleep(0.005);

		return (action);
	}

	/**
	 * Method saves to disk the best offer bid of these domain.
	 */
	@Override
	public void endSession(NegotiationResult result) {
		try {
			// Update the parameter we save to disk
			if (this.utilitySpace.getUtility(this.bestOffer) < this.utilitySpace
					.getUtility(this.maxBidRecieved)) {
				this.bestOffer = this.maxBidRecieved;
			}

			// Save it.
			this.saveSessionData(this.bestOffer);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Method calculates the next bid. It takes the maximum between the global
	 * maximum and a new randomized bid within a fixed number of iterations.
	 * 
	 * @return
	 */
	private Bid calculateNextBid() {
		try {
			Bid nextBid = this.generateBid();

			// Return the maximum bid between the global maximum and the new
			// randomized bid
			if (this.utilitySpace.getUtility(nextBid) > this.utilitySpace
					.getUtility(this.maxBid)) {
				this.maxBid = nextBid;
			}

			return (this.maxBid);
		} catch (Exception e) { // Return the max bid
			e.printStackTrace();
			return (this.maxBid);
		}
	}

	/**
	 * Method iterates fix number of times and returns the maximum bid within
	 * all generated random bids.
	 * 
	 * @return
	 * @throws Exception
	 */
	private Bid generateBid() throws Exception {
		double maxUtility = 0;
		Bid maxBid = null;

		for (int i = 0; i < this.MAX_ITERATIONS; i++) {
			Bid bid = this.utilitySpace.getDomain().getRandomBid(null);
			double utility = this.utilitySpace.getUtility(bid);

			// If its value higher than the maximum so far.
			if (utility > maxUtility) {
				maxUtility = utility;
				maxBid = bid;
			}
		}

		return (maxBid);
	}

	/**
	 * Method stores and returns all the best bids for us that are more than a
	 * given threshold.
	 * 
	 * @return
	 */
	private Map<Bid, Double> getBestBidsForUs() {
		Map<Bid, Double> bestBids = new HashMap<Bid, Double>();

		for (int i = 0; i < this.MAX_ITERATIONS; i++) {
			Bid randomBid = this.utilitySpace.getDomain().getRandomBid(null);
			try {
				double utility = this.utilitySpace.getUtility(randomBid);

				// If the value of the random bid is more than the utility
				// threshold, save it.
				if (utility >= this.UTILITY_THRESHOLD) {
					if (!bestBids.containsKey(randomBid)) {
						bestBids.put(randomBid, utility);
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return (bestBids);
	}

	@Override
	public String getDescription() {
		return "ANAC2012";
	}
}
