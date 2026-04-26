package agents.anac.y2018.sontag;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfoList;

/**
 * Sample party that accepts the Nth offer, where N is the number of sessions
 * this [agent-profile] already did.
 */
public class Sontag extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private int nrChosenActions = 0; // number of times chosenAction was called.
	private StandardInfoList history;
	private double discountFactor;
	private double reservationValue;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		discountFactor = info.getUtilitySpace().getDiscountFactor();
		reservationValue = info.getUtilitySpace().getReservationValue();

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();
	}

	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double t = this.getTimeLine().getTime();
		nrChosenActions++;

		if (getUtility(lastReceivedBid) >= getLowerBound(t)) {
			return new Accept(getPartyId(), lastReceivedBid);
		} else {
			if(nrChosenActions == 1) {
				try {
					return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			return new Offer(getPartyId(), generateBid(t));
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
		}
	}

	private double getUpperBound(double t) {
		return 1.0;
	}

	private double getLowerBound(double t) {
		return t / 2.5 - Math.log10(t / 2 + 0.1);
	}


	private Bid generateBid(double t) {
		return generateRandomBid(getUpperBound(t), getLowerBound(t));
	}

	private Bid generateRandomBid(double upperBound, double lowerBound) {
		Bid randomBid;
		double utility;
		do {
			randomBid = generateRandomBid();
			try {
				utility = utilitySpace.getUtility(randomBid);
			} catch (Exception e) {
				utility = 0.0;
			}
		}
		while (utility < lowerBound || upperBound < utility);
		return randomBid;
	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

}
