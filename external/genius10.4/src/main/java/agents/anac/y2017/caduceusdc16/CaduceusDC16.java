package agents.anac.y2017.caduceusdc16;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import agents.anac.y2016.atlas3.Atlas32016;
import agents.anac.y2016.caduceus.agents.Caduceus.UtilFunctions;
import agents.anac.y2016.farma.Farma;
import agents.anac.y2016.myagent.MyAgent;
import agents.anac.y2016.parscat.ParsCat;
import agents.anac.y2016.yxagent.YXAgent;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.utility.AbstractUtilitySpace;

/**
 * This is your negotiation party.
 */
public class CaduceusDC16 extends AbstractNegotiationParty {

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */

	public double discountFactor = 0; // if you want to keep the discount factor
	private double selfReservationValue = 0.75;
	private double percentageOfOfferingBestBid = 0.83;
	private Random random;
	private Bid lastReceivedBid = null;
	private AbstractUtilitySpace uspace = null;

	public NegotiationParty[] agents = new NegotiationParty[5];
	public double[] scores = UtilFunctions
			.normalize(new double[] { 5, 4, 3, 2, 1 });

	public double getScore(int agentIndex) {
		return scores[agentIndex];
	}

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		random = new Random(info.getRandomSeed());

		agents[0] = new YXAgent();
		agents[1] = new ParsCat();
		agents[2] = new Farma();
		agents[3] = new MyAgent();
		agents[4] = new Atlas32016();

		uspace = getUtilitySpace();
		discountFactor = getUtilitySpace().getDiscountFactor(); // read
																		// discount
																		// factor
		double reservationValue = getUtilitySpace()
				.getReservationValueUndiscounted();

		System.out.println("Discount Factor is " + discountFactor);
		System.out.println("Reservation Value is " + reservationValue);

		percentageOfOfferingBestBid = percentageOfOfferingBestBid
				* discountFactor;
		StandardInfoList history = (StandardInfoList) getData().get();
		if (!history.isEmpty()) {
			double total = 0;
			for (StandardInfo prevHist : history) {
				int numberOfAgents = prevHist.getAgentProfiles().size();
				List<genius.core.list.Tuple<String, Double>> agentUtilities = prevHist
						.getUtilities();
				int agentUtilitySize = agentUtilities.size();
				List<genius.core.list.Tuple<String, Double>> finalUtilities = agentUtilities
						.subList(agentUtilitySize - numberOfAgents,
								agentUtilitySize);
				for (genius.core.list.Tuple<String, Double> agentUtility : finalUtilities) {
					if (agentUtility.get1().toLowerCase()
							.contains("CaduceusDC16".toLowerCase())) {
						total += agentUtility.get2();
					}
				}
			}
			selfReservationValue = total / history.size();
			System.out.println(selfReservationValue);
		}

		for (NegotiationParty agent : agents) {
			agent.init(info);
		}

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		if (isBestOfferTime()) {
			Bid bestBid = this.getBestBid();
			if (bestBid != null)
				return new Offer(getPartyId(), bestBid);
			else
				System.err.println("Best Bid is null?");
		}

		ArrayList<Bid> bidsFromAgents = new ArrayList<Bid>();
		ArrayList<Action> possibleActions = new ArrayList<Action>();

		for (NegotiationParty agent : agents) {
			Action action = agent.chooseAction(validActions);
			possibleActions.add(action);
		}

		double scoreOfAccepts = 0;
		double scoreOfBids = 0;
		ArrayList<Integer> agentsWithBids = new ArrayList<>();

		int i = 0;
		for (Action action : possibleActions) {
			if (action instanceof Accept) {
				scoreOfAccepts += getScore(i);
			} else if (action instanceof Offer) {
				scoreOfBids += getScore(i);
				bidsFromAgents.add(((Offer) action).getBid());
				agentsWithBids.add(i);
			}
			i++;
		}
		if (scoreOfAccepts > scoreOfBids
				&& uspace.getUtility(lastReceivedBid) >= selfReservationValue) {
			return new Accept(getPartyId(), lastReceivedBid);

		} else if (scoreOfBids > scoreOfAccepts) {
			return new Offer(getPartyId(), getMostProposedBidWithWeight(
					agentsWithBids, bidsFromAgents));
		}

		return new Offer(getPartyId(), getBestBid());
	}

	private Bid getRandomizedAction(ArrayList<Integer> agentsWithBids,
			ArrayList<Bid> bidsFromAgents) {
		double[] possibilities = new double[agentsWithBids.size()];

		int i = 0;
		for (Integer agentWithBid : agentsWithBids) {
			possibilities[i] = getScore(agentWithBid);
			i++;
		}
		possibilities = UtilFunctions.normalize(possibilities);
		UtilFunctions.print(possibilities);
		double randomPick = random.nextDouble();

		double acc = 0;
		i = 0;
		for (double possibility : possibilities) {
			acc += possibility;

			if (randomPick < acc) {
				return bidsFromAgents.get(i);
			}

			i++;
		}

		return null;
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action. Can be null.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer)
			lastReceivedBid = ((Offer) action).getBid();

		for (NegotiationParty agent : agents) {
			agent.receiveMessage(sender, action);
		}

	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

	private Bid getBestBid() {
		try {
			return this.utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private boolean isBestOfferTime() {
		return this.getTimeLine()
				.getCurrentTime() < (this.getTimeLine().getTotalTime()
						* percentageOfOfferingBestBid);
	}

	private Bid getMostProposedBidWithWeight(ArrayList<Integer> agentsWithBids,
			ArrayList<Bid> bidsFromAgents) {
		try {
			List<Issue> allIssues = bidsFromAgents.get(0).getIssues();
			HashMap<Integer, Value> bidMap = new HashMap<>();
			for (int i = 1; i <= allIssues.size(); i++) {
				Map<Value, Double> proposedValues = new HashMap<>();
				for (int k = 0; k < agentsWithBids.size(); k++) {
					Value agentBidValue = bidsFromAgents.get(k).getValue(i);
					int agentNumber = agentsWithBids.get(k);
					Double val = proposedValues.get(agentBidValue);
					proposedValues.put(agentBidValue, val == null ? 1
							: val + scores[agentsWithBids.get(k)]);
				}
				Map.Entry<Value, Double> max = null;

				for (Map.Entry<Value, Double> e : proposedValues.entrySet()) {
					if (max == null || e.getValue() > max.getValue())
						max = e;
				}
				bidMap.put(i, max.getKey());
			}
			return new Bid(utilitySpace.getDomain(), bidMap);

		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
}