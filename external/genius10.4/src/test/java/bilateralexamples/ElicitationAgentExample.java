package bilateralexamples;

import java.util.List;
import java.util.Random;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * This class implements an example of an agent that uses elicitation in its strategy under uncertainty.
 * Note that the agent works exclusively under uncertainty.
 * @author Adel Magra
 *
 */

@SuppressWarnings("serial")
public class ElicitationAgentExample extends AbstractNegotiationParty {

	private Random random = new Random();
	/**
	 * Initializes a new instance of the agent.
	 */
	@Override
	public void init(NegotiationInfo info) 
	{
		super.init(info);
		if (!hasPreferenceUncertainty())
		{
			log("There is no preference uncertainty. Try this agent with a negotiation scenario that has preference uncertainty enabled.");
			return;
		}
	}

	@Override
	
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		
		List<Bid> bidOrder = userModel.getBidRanking().getBidOrder();
		
		if (getLastReceivedAction() instanceof Offer) {
			Bid receivedBid = ((Offer) getLastReceivedAction()).getBid();
			//Elicit receivedBid if it is not in bidOrder and TBC < 0.15
			if(!bidOrder.contains(receivedBid) && user.getTotalBother()<0.15) {
				userModel = user.elicitRank(receivedBid,userModel);
				bidOrder = userModel.getBidRanking().getBidOrder();
			}
			//Accept if and only if received bid is in top 10% of known bids 
			double percentile = (bidOrder.size() - bidOrder.indexOf(receivedBid)) / (double) bidOrder.size();
			if (percentile < 0.1)
				return new Accept(getPartyId(), receivedBid);
			
			//Otherwise, offer a random bid in the top 10% of known bids
			int threshold = (int) (0.9*bidOrder.size());
			List<Bid> topList = bidOrder.subList(threshold, bidOrder.size());
			int index = random.nextInt(topList.size());
			return new Offer(getPartyId(), topList.get(index));
		}
		//First action
		return new Offer(getPartyId(),bidOrder.get(bidOrder.size()-1));
	}

	@Override
	public String getDescription() {
		return "Elicitation Agent Example";
	}

	private void log(String string) {
		System.out.println(string);
	}

	
}

