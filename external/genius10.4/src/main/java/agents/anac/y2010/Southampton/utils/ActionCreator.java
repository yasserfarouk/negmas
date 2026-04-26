package agents.anac.y2010.Southampton.utils;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;

/**
 * This factory class has been created to allow Actions to be constructed.
 * 
 * @author Colin Williams
 * 
 */
public class ActionCreator {

	public static Action createOffer(Agent agent, Bid bid) {
		return new Offer(agent.getAgentID(), bid);
	}

	public static Action createAccept(Agent agent, Bid oppBid) {
		return new Accept(agent.getAgentID(), oppBid);
	}

}
