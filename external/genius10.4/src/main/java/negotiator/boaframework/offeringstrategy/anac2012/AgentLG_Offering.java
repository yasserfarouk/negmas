package negotiator.boaframework.offeringstrategy.anac2012;

import java.util.Map;

import agents.anac.y2012.AgentLG.OpponentBids;
import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2012.AgentLGSAS;

public class AgentLG_Offering extends OfferingStrategy {

	private Bid myLastBid = null;
	private OpponentBids oppenentsBid;
	private boolean bidLast = false;

	public AgentLG_Offering() {
	}

	public AgentLG_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) throws Exception {
		init(negoSession, model, oms, null);
	}

	/**
	 * Init required for the Decoupled Framework.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		oppenentsBid = new OpponentBids(negoSession.getUtilitySpace());
		negotiationSession = negoSession;
		helper = new AgentLGSAS(negotiationSession, oppenentsBid, model, oms);
	}

	@Override
	public BidDetails determineOpeningBid() {
		if (!negotiationSession.getOpponentBidHistory().isEmpty())
			oppenentsBid.addBid(negotiationSession.getOpponentBidHistory().getLastBid());
		myLastBid = negotiationSession.getMaxBidinDomain().getBid();
		return negotiationSession.getMaxBidinDomain();
	}

	@Override
	public BidDetails determineNextBid() {

		oppenentsBid.addBid(negotiationSession.getOpponentBidHistory().getLastBid());

		BidDetails currentAction = null;
		try {
			double time = negotiationSession.getTime();

			if (bidLast) {
				// System.out.println("Decoupled Last Bid");
				currentAction = negotiationSession.getOwnBidHistory().getLastBidDetails();
			}
			// there is lot of time ->learn the opponent and bid the 1/4 most
			// optimal bids
			else if (time < 0.6) {
				if (!negotiationSession.getOpponentBidHistory().isEmpty())
					currentAction = ((AgentLGSAS) helper).getNextOptimicalBid(time);
				myLastBid = negotiationSession.getOwnBidHistory().getLastBid();
			} else {
				// the time is over -> bid the opponents max utility bid for me
				if (time >= 0.9995) {
					myLastBid = negotiationSession.getOpponentBidHistory().getBestBidDetails().getBid();
					if (negotiationSession.getUtilitySpace().getUtilityWithDiscount(myLastBid,
							time) < negotiationSession.getUtilitySpace().getReservationValueWithDiscount(time))
						myLastBid = ((AgentLGSAS) helper).getMyminBidfromBids();
					currentAction = new BidDetails(myLastBid,
							negotiationSession.getUtilitySpace().getUtility(myLastBid));
				} else {
					// Comprise and chose better bid for the opponents that
					// still good for me
					currentAction = ((AgentLGSAS) helper).getNextBid(time);
				}
			}
			if (oppenentsBid.getOpponentsBids().contains(currentAction.getBid())) {
				bidLast = true;
			}

		} catch (Exception e) {
			System.out.println("Error and thus accept: " + e);
			// currentAction = new Accept(getAgentID());
		}

		return currentAction;
	}

	@Override
	public String getName() {
		return "2012 - AgentLG";
	}
}