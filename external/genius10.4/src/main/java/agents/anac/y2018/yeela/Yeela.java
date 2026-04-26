package agents.anac.y2018.yeela;

import java.util.List;

import java.util.Vector;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * group1.Agent1 returns the bid that maximizes its own utility for half of the negotiation session.
 * In the second half, it offers a random bid. It only accepts the bid on the table in this phase,
 * if the utility of the bid is higher than Example Agent's last bid.
 */
public class Yeela extends AbstractNegotiationParty {
    /**
	 * 
	 */
	private static final long serialVersionUID = -2676016971703971492L;

	private final String description = "yeela Agent";

    private Bid lastReceivedOffer; // offer on the table
    private Bid bestReceivedOffer; // best offer seen
    private Learner curLearner;
    private boolean firstGameAct;
    private double timeToGiveUp = 0.75;
    private List<Bid> bids;
    private NegotiationInfo m_info;
    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        System.out.println("Discount Factor is " + info.getUtilitySpace().getDiscountFactor());
        System.out.println("Reservation Value is " + info.getUtilitySpace().getReservationValueUndiscounted());
        
        firstGameAct = true;
        m_info = info;
        bids = new Vector<Bid>();
        
		curLearner = new Learner(getMaxUtilityBid(), info);
    }

    /**
     * When this function is called, it is expected that the Party chooses one of the actions from the possible
     * action list and returns an instance of the chosen action.
     *
     * @param list
     * @return
     */
    public Action chooseAction(List<Class<? extends Action>> list) {
        // According to Stacked Alternating Offers Protocol list includes
        // Accept, Offer and EndNegotiation actions only.
        double time = getTimeLine().getTime(); // Gets the time, running from t = 0 (start) to t = 1 (deadline).
                                               // The time is normalized, so agents need not be
                                               // concerned with the actual internal clock.

        System.out.println(time);

        // if we are first
        if (firstGameAct)
        {
        	firstGameAct = false;
        	bids.add(this.getMaxUtilityBid());
        	return new Offer(this.getPartyId(), this.getMaxUtilityBid());
        }
        
        try
        {
        	if (timeToGiveUp < time)
        	{
        		return new Accept(this.getPartyId(), lastReceivedOffer);
        	}

        	// create new offer
	        bids.add(curLearner.run(lastReceivedOffer));

	        // decide whether to accept counter offer
	        for (Bid bid : bids)
	        {
		        if (m_info.getUtilitySpace().getUtility(lastReceivedOffer) == m_info.getUtilitySpace().getUtility(bid))
	        	{
	        		return new Accept(this.getPartyId(), lastReceivedOffer);
	        	}
	        }

	        // decide whether to offer previous counter offer since its better than newly suggested offer
	        if (m_info.getUtilitySpace().getUtility(bestReceivedOffer) > m_info.getUtilitySpace().getUtility(bids.get(bids.size() - 1)))
	        {
	        	return new Offer(this.getPartyId(), bestReceivedOffer);
	        }

	        // suggest our new offer
	        return new Offer(this.getPartyId(), bids.get(bids.size() - 1));
        }
        catch (Exception e)
		{
			e.printStackTrace();
        	return new Accept(this.getPartyId(), lastReceivedOffer);
        }
    }

    /**
     * This method is called to inform the party that another NegotiationParty chose an Action.
     * @param sender
     * @param act
     */
    @Override
    public void receiveMessage(AgentID sender, Action act) {
        super.receiveMessage(sender, act);

        if (act instanceof Offer) { // sender is making an offer
            Offer offer = (Offer) act;
            
            // storing last received offer
            lastReceivedOffer = offer.getBid();
            
            if ((null == bestReceivedOffer) || (m_info.getUtilitySpace().getUtility(bestReceivedOffer) < m_info.getUtilitySpace().getUtility(lastReceivedOffer)))
            {
            	bestReceivedOffer = lastReceivedOffer;
            }
            firstGameAct = false;
        }
    }

    /**
     * A human-readable description for this party.
     * @return
     */
    @Override
    public String getDescription() {
        return "ANAC2018";
    }

    private Bid getMaxUtilityBid() {
        try
        {
            return this.utilitySpace.getMaxUtilityBid();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
