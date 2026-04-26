package agents.anac.y2016.caduceus;

import java.util.List;

import java.util.ArrayList;
import java.util.Random;

import agents.anac.y2016.caduceus.agents.Atlas3.Atlas3;
import agents.anac.y2016.caduceus.agents.Caduceus.UtilFunctions;
import agents.anac.y2016.caduceus.agents.ParsAgent.ParsAgent;
import agents.anac.y2016.caduceus.agents.RandomDance.RandomDance;
import agents.anac.y2016.caduceus.agents.kawaii.kawaii;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.persistent.DefaultPersistentDataContainer;
import genius.core.persistent.PersistentDataType;

/**
 * This is your negotiation party.
 */
public class Caduceus extends AbstractNegotiationParty {

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

	public NegotiationParty[] agents = new NegotiationParty[5];
	public double[] scores = UtilFunctions
			.normalize(new double[] { 500, 10, 5, 3, 1 });

	public double getScore(int agentIndex) {
		return scores[agentIndex];
	}

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		random = new Random(info.getRandomSeed());

		agents[0] = new ParsAgent();
		agents[1] = new RandomDance();
		agents[2] = new kawaii();
		agents[3] = new Atlas3();
		agents[4] = new agents.anac.y2016.caduceus.agents.Caduceus.Caduceus();

		discountFactor = getUtilitySpace().getDiscountFactor(); // read
																		// discount
																		// factor
		double reservationValue = getUtilitySpace()
				.getReservationValueUndiscounted();

		System.out.println("Discount Factor is " + discountFactor);
		System.out.println("Reservation Value is " + reservationValue);

		selfReservationValue = Math.max(selfReservationValue, reservationValue);
		percentageOfOfferingBestBid = percentageOfOfferingBestBid
				* discountFactor;

		for (NegotiationParty agent : agents) {

			agent.init(new NegotiationInfo(getUtilitySpace(), info.getUserModel(),null,
					info.getDeadline(), info.getTimeline(),
					info.getRandomSeed(), info.getAgentID(),
					new DefaultPersistentDataContainer(null,
							PersistentDataType.DISABLED)));
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
		if (scoreOfAccepts > scoreOfBids) {
			return new Accept(getPartyId(),
					((ActionWithBid) getLastReceivedAction()).getBid());

		} else if (scoreOfBids > scoreOfAccepts) {
			return new Offer(getPartyId(),
					getRandomizedAction(agentsWithBids, bidsFromAgents));
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

		for (NegotiationParty agent : agents) {
			agent.receiveMessage(sender, action);
		}

	}

	@Override
	public String getDescription() {
		return "ANAC2016";
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
}
