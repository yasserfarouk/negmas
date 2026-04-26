package agents.anac.y2015.pnegotiator;

import java.util.ArrayList;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.issue.Objective;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is your negotiation party.
 */
public class PNegotiator extends AbstractNegotiationParty {

	// State of the agent (am I being a hardliner or am I conceding?)
	public enum AgentState {
		HARDLINER, CONCEDER
	};

	public AgentState agentState;

	// Parameters of the negotiation
	// private UtilitySpace utilitySpace;
	// private Timeline timeline;
	private List<Objective> objectives;

	// Other private fields
	private Bid currentBid = null;
	private Bid nextBid = null;
	// private double concessionFactor = 1;
	// private int totalBids = 1;

	private BestBids bestBids;
	private BayesLogic bayesLogic;

	private double lRng = 0.9, rRng = 1.0;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		// Set agent's game parameters
		// this.utilitySpace = utilitySpace;
		// this.timeline = timeline;
		this.objectives = utilitySpace.getDomain().getObjectives();

		try {
			this.bestBids = new BestBids(
					(AdditiveUtilitySpace) this.utilitySpace);
		} catch (Exception e) {
			e.printStackTrace();
		}

		agentState = AgentState.HARDLINER;

		// Print stuff for the benefit of our understanding
		// System.out.println("Discount factor: " +
		// utilitySpace.getDiscountFactor());
		// for(int i = 1; i < objectives.size(); ++i)
		// {
		// System.out.println("(Name, Type, Weight): (" +
		// objectives.get(i).getName() + ", " + objectives.get(i).getType() +
		// ", "+ utilitySpace.getWeight(i) + ")");
		// }
		//
		// System.out.println("Disagreement point (Discounted): " +
		// utilitySpace.getReservationValueWithDiscount(timeline));
		// System.out.println("Disagreement point (Undiscounted): " +
		// utilitySpace.getReservationValueUndiscounted());
		// System.out.println("Discount factor: " +
		// utilitySpace.getDiscountFactor());
		// System.out.println("Number of possible bids: " +
		// domain.getNumberOfPossibleBids());
	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		// ++totalBids;
		try {
			setState();
			updateBayes();
			System.out.println(currentBid);
			if (currentBid == null) {
				nextBid = utilitySpace.getMaxUtilityBid();
			} else {
				// setState();
				lRng = getConcessionUtility();
				Bid bestBid = bestBids.getRandomBid(generateRandomBid(), rand,
						lRng, rRng);
				double CU = utilitySpace.getUtility(currentBid);
				System.out.format("%6.4f < %6.4f < %6.4f\n", lRng,
						utilitySpace.getUtility(bestBid), rRng);
				switch (agentState) {
				case HARDLINER:
					// System.out.println("HARDLINER");
					nextBid = bestBid;
					break;
				default:
					// System.out.println("CONCEDER");
					Bid bayes;
					double U1;
					do {
						bayes = bayesLogic
								.bayesBid(utilitySpace.getMaxUtilityBid());
						U1 = utilitySpace.getUtility(bayes);
						// System.out.format("V: %3d U: %6.4f\n", bayesLogic.V,
						// U1);
						bayesLogic.V++;
					} while (U1 < lRng);
					bayesLogic.V = 1;
					// bayes = bayesLogic.bayesBid2(rand);
					// U1 = utilitySpace.getUtility(bayes);
					// System.out.format("Bayes: %6.4f | Proposed: %6.4f\n",
					// U1,CU);
					nextBid = bayes;
				}
				// bayesLogic.updateOpponentFrequency(bestBid, 0);
				if (utilitySpace.getUtility(currentBid) >= lRng)
					return new Accept(getPartyId(), currentBid);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		// Print the total utility of the bid we are making and submit it
		System.out.println("Proposing bid with utility: "
				+ utilitySpace.getUtilityWithDiscount(nextBid, timeline));
		return new Offer(getPartyId(), nextBid);
	}

	private void updateBayes() throws Exception {
		if (bayesLogic == null) {
			this.bayesLogic = new BayesLogic(
					(AdditiveUtilitySpace) this.utilitySpace,
					this.getNumberOfParties());
		}
		bayesLogic.T = (int) (getTimeLine().getTime() * 100);
		System.out.format("T: %d\n", bayesLogic.T);
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (!players.contains(sender)) {
			players.add(sender);
		}
		int P = players.indexOf(sender);
		if (DefaultAction.getBidFromAction(action) != null)
			currentBid = DefaultAction.getBidFromAction(action);
		try {
			if (currentBid != null) {
				updateBayes();
				bayesLogic.updateOpponentFrequency(currentBid, P);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	ArrayList<Object> players = new ArrayList<Object>();

	// A dumb function that just sets a scalar concession factor based on the
	// current time
	private double getConcessionUtility() throws Exception {
		double Bj = utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
		double Cj = Bj * timeline.getTime() * timeline.getTime();
		double Uj = Bj - (Cj / objectives.size());
		// System.out.println("Max : "+Bj+"Cj "+Cj+" time:
		// "+timeline.getTime()+"Bound:"+Uj);
		return Uj * Uj;
	}

	// Logic to decide the next bid
	// private void decideNextBid() throws Exception {
	// switch(agentState) {
	// case HARDLINER:
	// System.out.println("MaxRand");
	// nextBid = bestBids.getRandomBid(generateRandomBid(), rand, 0.8, 1.);
	// break;
	// case CONCEDER:
	// System.out.println("Bayes");
	// nextBid = bayesLogic.bayesBid(utilitySpace.getMaxUtilityBid());
	// }
	// }

	private void setState() {
		if (timeline.getTime() >= .2) {
			agentState = AgentState.CONCEDER;
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}
}
