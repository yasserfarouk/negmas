package agents.anac.y2019.podagent;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.BidHistory;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.timeline.DiscreteTimeline;

public class Group1_AS extends AcceptanceStrategy {
	public Group1_AS() {
	}

	public Group1_AS(NegotiationSession negoSession, OfferingStrategy strat, double friendliness, double panic) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.opponentModel = opponentModel;
	}
	
	/**
	 * Determine whether to accept or not. We use a standard metric of comparing the offer to our next bid, 
	 * but because out agent uniquely relies so heavily on our bidding strategy we only give up when we are out of turns 
	 * without the need for complicated methods of acceptance.
	 * 
	 * @return Actions.Reject if our BS is not in panicMode, 
	 * or we are contesting a hard-headed opponent below a threshold.
	 * @return Actions.Accept if we are out of turns, our next bid is lower,
	 * or a hard-headed opponent beats a threshold.
	 */
	@Override
	public Actions determineAcceptability() {

		if (negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil() >= offeringStrategy
				.getNextBid().getMyUndiscountedUtil()) {
			return Actions.Accept;
		}
		if (!getPanicMode()) {
			return Actions.Reject;
		}
		boolean hardHead = false;
		if (opponentModel instanceof Group1_OM) {
			hardHead = ((Group1_OM) opponentModel).isHardHeaded();
		}
		if(hardHead) {
			if(negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil() >= 0.95) {
				return Actions.Accept;
			}
			return Actions.Reject;
		}
		return Actions.Reject;
	}
	
	/**
	 * Get PanicMode from Bidding Strategy
	 * 
	 * @return boolean in PanicMode or not
	 */
	private Boolean getPanicMode() {
		if(offeringStrategy instanceof Group1_BS) {
			return ((Group1_BS) offeringStrategy).getPanicMode();
		}
		else {
				return false;
			}
		}

	@Override
	public String getName() {
		return "Group1_AS";
	}
}
