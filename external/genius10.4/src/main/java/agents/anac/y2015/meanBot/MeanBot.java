package agents.anac.y2015.meanBot;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * This is your negotiation party.
 */
public class MeanBot extends AbstractNegotiationParty {

	private Bid currentBid;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
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
		try {
			if (validActions.contains(Accept.class) && timeline.getTime() >= .95
					&& utilitySpace.getUtility(currentBid) > .5) {
				return new Accept(getPartyId(), currentBid);
			} else
				return new Offer(getPartyId(), utilitySpace.getMaxUtilityBid());

		} catch (Exception e) {
			return new Accept(getPartyId(), currentBid); // Not sure what to put
															// here...
			// Don't know what
			// error I would be hitting.
		}
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

		if (action instanceof Offer) {
			// Know what the most recent bid on the table is, for evaluation.
			currentBid = DefaultAction.getBidFromAction(action);
		}
		// Here you can listen to other parties' messages
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}
