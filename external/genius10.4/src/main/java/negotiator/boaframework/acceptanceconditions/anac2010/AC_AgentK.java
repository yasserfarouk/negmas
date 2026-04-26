package negotiator.boaframework.acceptanceconditions.anac2010;

import java.util.Map;
import java.util.Random;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2010.AgentKSAS;

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
public class AC_AgentK extends AcceptanceStrategy {

	private Random random100;
	private boolean activeHelper = false;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_AgentK() {
	}

	public AC_AgentK(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
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

		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("AgentK"))) {
			helper = new AgentKSAS(negotiationSession);
			activeHelper = true;
		} else {
			helper = (AgentKSAS) offeringStrategy.getHelper();
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
				p = ((AgentKSAS) helper).calculateAcceptProbability();
			} else {
				p = ((AgentKSAS) helper).getAcceptProbability();
			}
			if (p > random100.nextDouble()) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2010 - AgentK";
	}
}