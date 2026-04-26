package parties.AlternatingMultipleOffers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.actions.OfferForVoting;
import genius.core.actions.Reject;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.protocol.AlternatingMultipleOffersProtocol;
import genius.core.protocol.DefaultMultilateralProtocol;

public class RandomAmopParty extends AbstractNegotiationParty {

	private final static float P_ACCEPT = 0.3f;

	private List<Bid> receivedBids = new ArrayList<Bid>();
	private Random rnd = new Random();

	enum Phase {
		OFFER, VOTE
	};

	private Phase phase = Phase.OFFER;

	/**
	 * We need to keep track of the number of votes we already did.
	 */
	private int votingIndex = 0;

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		if (possibleActions.contains(OfferForVoting.class)) {
			Bid bid = generateRandomBid();
			addOffer(bid);
			return new OfferForVoting(getPartyId(), bid);
		}
		if (possibleActions.contains(Accept.class)) {
			setPhase(Phase.VOTE);
			if (votingIndex >= receivedBids.size()) {
				throw new IllegalStateException("Received more requests for vote (" + (votingIndex + 1)
						+ ") than number of received offers (" + receivedBids.size() + ")!");
			}
			Bid bid = receivedBids.get(votingIndex++);

			if (rnd.nextFloat() < P_ACCEPT) {
				return new Accept(getPartyId(), bid);
			} else {
				return new Reject(getPartyId(), bid);
			}
		}
		throw new IllegalStateException("Unknown action request " + possibleActions);
	}

	/**
	 * Try to track which phase we are in...
	 * 
	 * @param newPhase
	 *            the phase that we supposedly are in now.
	 */
	private void setPhase(Phase newPhase) {
		if (phase != newPhase) {
			phase = newPhase;
			if (phase == Phase.OFFER) {
				// entered new offer round
				if (receivedBids.size() != votingIndex) {
					System.out.println("Warning: entered new Offer round but we voted only on " + votingIndex
							+ " of the " + receivedBids);
				}
				votingIndex = 0;
				receivedBids.clear();
			}
		}
	}

	private void addOffer(Bid bid) {
		setPhase(Phase.OFFER);
		receivedBids.add(bid);
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		if (action instanceof OfferForVoting) {
			addOffer(((Offer) action).getBid());
		} else if (action instanceof Accept || action instanceof Reject) {
			setPhase(Phase.VOTE);
		} else {
			System.out.println("Warning: ignoring unknown message " + action);
		}
	}

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return AlternatingMultipleOffersProtocol.class;
	}

	@Override
	public String getDescription() {
		return "Random AMOP Party";
	}

}
