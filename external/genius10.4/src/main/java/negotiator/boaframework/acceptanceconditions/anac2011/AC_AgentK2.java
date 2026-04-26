package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.Map;
import java.util.Random;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.AgentK2SAS;

/**
 * This is the decoupled Acceptance Condition from Agent K (ANAC2010). The code
 * was taken from the ANAC2010 Agent K and adapted to work within the BOA
 * framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 *
 * @author Mark Hendrikx
 * @version 25/12/11
 */
public class AC_AgentK2 extends AcceptanceStrategy {

	private Random random100;
	private boolean activeHelper = false;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_AgentK2() {
	}

	public AC_AgentK2(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		initializeAgent(negoSession, strat);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		initializeAgent(negoSession, strat);
	}

	public void initializeAgent(NegotiationSession negotiationSession, OfferingStrategy os) throws Exception {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = os;

		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("AgentK2"))) {
			helper = new AgentK2SAS(negotiationSession);
			activeHelper = true;
		} else {
			helper = (AgentK2SAS) offeringStrategy.getHelper();
		}

		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
		} else {
			random100 = new Random();
		}
	}

	@Override
	public Actions determineAcceptability() {
		BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		if (opponentBid != null) {
			double p;
			if (activeHelper) {
				p = ((AgentK2SAS) helper).calculateAcceptProbability();
			} else {
				p = ((AgentK2SAS) helper).getAcceptProbability();
			}
			if (p > random100.nextDouble()) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2011 - AgentK2";
	}
}