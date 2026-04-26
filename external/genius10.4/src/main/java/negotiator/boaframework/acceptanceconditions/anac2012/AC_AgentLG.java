package negotiator.boaframework.acceptanceconditions.anac2012;

import java.util.Map;

import agents.anac.y2012.AgentLG.OpponentBids;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2012.AgentLGSAS;

public class AC_AgentLG extends AcceptanceStrategy {

	private boolean activeHelper = false;
	private OpponentBids opponentsBid;

	public AC_AgentLG() {
	}

	public AC_AgentLG(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;

		// checking if offeringStrategy SAS is a AgentLGSAS
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("AgentLR"))) {
			opponentsBid = new OpponentBids(negoSession.getUtilitySpace());
			helper = new AgentLGSAS(negotiationSession, opponentsBid, new NoModel(), null);
			activeHelper = true;
		} else {
			helper = (AgentLGSAS) offeringStrategy.getHelper();
		}
	}

	@Override
	public Actions determineAcceptability() {

		if (activeHelper) {
			opponentsBid.addBid(negotiationSession.getOpponentBidHistory().getLastBid());
		}

		double time = negotiationSession.getTime();
		if (negotiationSession.getOwnBidHistory().isEmpty()) {
			return Actions.Reject;
		}
		double myUtility = negotiationSession.getUtilitySpace()
				.getUtilityWithDiscount(negotiationSession.getOwnBidHistory().getLastBidDetails().getBid(), time);

		double opponentUtility = negotiationSession.getUtilitySpace()
				.getUtilityWithDiscount(negotiationSession.getOpponentBidHistory().getLastBid(), time);

		/*
		 * if(activeHelper){ if(!(time<0.6)) { if ( !(time>=0.9995)){ //to set
		 * some parameters ((AgentLRSAS) helper).getNextBid(time); }
		 * 
		 * } }
		 */

		// System.out.println("decoupled Condition 1: " + (opponentUtility >=
		// myUtility*0.99));
		// System.out.println("decoupled Condition 2: " + ( time>0.999 &&
		// opponentUtility >= myUtility*0.9));
		// System.out.println("decoupled Condition 3: " + (((AgentLRSAS)
		// helper).getMyBidsMinUtility(time)<= opponentUtility));

		// accept if opponent offer is good enough or there is no time and the
		// offer is 'good'
		if (opponentUtility >= myUtility * 0.99 || (time > 0.999 && opponentUtility >= myUtility * 0.9)
				|| ((AgentLGSAS) helper).getMyBidsMinUtility(time) <= opponentUtility) {
			return Actions.Accept;

		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2012 - AgentLG";
	}

}
