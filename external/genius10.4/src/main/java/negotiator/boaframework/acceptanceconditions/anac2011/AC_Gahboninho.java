package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.GahboninhoSAS;

/**
 * @author Mark Hendrikx and Alex Dirkzwager
 * 
 *         This is the decoupled Acceptance Conditions from the Gahboninho
 *         (ANAC2011). The code was taken from the ANAC2011 Gahboninho and
 *         adapted to work within the BOA framework.
 * 
 *         Decoupling Negotiating Agents to Explore the Space of Negotiation
 *         Strategies T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M.
 *         Jonker
 * 
 *         BUG In the original version the opponent model is only updated once,
 *         which is good for the performance of the agent, as the model is very
 *         slow.
 */
public class AC_Gahboninho extends AcceptanceStrategy {

	private boolean activeHelper = false;
	private boolean done = false;

	public AC_Gahboninho() {
	}

	public AC_Gahboninho(NegotiationSession negotiationSession, OfferingStrategy offeringStrategy) {
		initializeAgent(negotiationSession, offeringStrategy);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		System.out.println("offering: " + strat);

		initializeAgent(negoSession, strat);

	}

	public void initializeAgent(NegotiationSession negoSession, OfferingStrategy strat) {
		this.negotiationSession = negoSession;
		System.out.println("negotiationSession: " + negotiationSession);

		this.offeringStrategy = strat;
		if (offeringStrategy.getHelper() != null && offeringStrategy.getHelper().getName().equals("Gahboninho")) {
			helper = offeringStrategy.getHelper();

		} else {
			helper = new GahboninhoSAS(negotiationSession);
			activeHelper = true;
		}
	}

	@Override
	public Actions determineAcceptability() {
		BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();

		if (activeHelper) {
			if (negotiationSession.getOpponentBidHistory().getHistory().size() < 2) {
				try {

					((GahboninhoSAS) helper).getIssueManager().ProcessOpponentBid(opponentBid.getBid());
					((GahboninhoSAS) helper).getOpponentModel().UpdateImportance(opponentBid.getBid());
				} catch (Exception e) {
					e.printStackTrace();
				}

			} else {
				try {
					if (!done) {
						((GahboninhoSAS) helper).getIssueManager().learnBids(opponentBid.getBid());
						done = true;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}

		if (((GahboninhoSAS) helper).getFirstActions() > 0 && opponentBid != null
				&& opponentBid.getMyUndiscountedUtil() > 0.95) {
			return Actions.Accept;
		}

		if (opponentBid != null && opponentBid.getMyUndiscountedUtil() >= ((GahboninhoSAS) helper).getIssueManager()
				.getMinimumUtilForAcceptance()) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2011 - Gahboninho";
	}
}
