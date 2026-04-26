package multipartyexample;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * This is your negotiation party.
 */
public class Groupn extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		System.out.println("Discount Factor is " + getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is " + getUtilitySpace().getReservationValueUndiscounted());

		// if you need to initialize some variables, please initialize them
		// below

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		// with 50% chance, counter offer
		// if we are the first party, also offer.
		if (lastReceivedBid == null || !validActions.contains(Accept.class) || Math.random() > 0.5) {
			return new Offer(getPartyId(), generateRandomBid());
		} else {
			return new Accept(getPartyId(), lastReceivedBid);
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
		}
	}

	@Override
	public String getDescription() {
		return "example party group N";
	}

}
