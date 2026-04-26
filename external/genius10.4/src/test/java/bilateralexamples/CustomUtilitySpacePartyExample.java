package bilateralexamples;

import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.DiscreteTimeline;
import genius.core.timeline.Timeline.Type;
import genius.core.uncertainty.BidRanking;
import genius.core.uncertainty.ExperimentalUserModel;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.CustomUtilitySpace;

/**
 * Example of a party that deals with preference uncertainty by defining a custom UtilitySpace 
 * based on the closest known bid.
 * 
 * Given a bid b and a preference ranking o1 <= o2 <= ... < on from the user model, it does the following:
 * It finds the outcome oi that is 'most similar' to b (in terms of overlapping values)
 * It then estimates u(b) to be: (i / n)  * (highestUtil - lowestUtil) 
 * 
 * Note that this agent's estimate of the utility function is not linear additive.
 * 
 */
@SuppressWarnings("serial")
public class CustomUtilitySpacePartyExample extends AbstractNegotiationParty 
{
	/** This agent will make bids above the minimuTarget */
	private double minimumTarget = 1;
	
	@Override
	public void init(NegotiationInfo info) 
	{
		super.init(info);
		log("This is an example of a party that deals with preference uncertainty by defining a Custom UtilitySpace estimate.");
		log("The user model is: " + userModel);
		if (!hasPreferenceUncertainty())
		{
			log("There is no preference uncertainty. Try this agent with a negotiation scenario that has preference uncertainty enabled.");
			return;
		}
		
		log("Lowest util: " + userModel.getBidRanking().getLowUtility() 
	    + ". Highest util: " + userModel.getBidRanking().getHighUtility());
		log("The estimated utility space is: " + getUtilitySpace());

		Bid randomBid = getUtilitySpace().getDomain().getRandomBid(rand);
		log("The estimate of the utility of a random bid (" + randomBid	+ ") is: " + getUtility(randomBid));

		if (userModel instanceof ExperimentalUserModel) 
		{
			log("You have given the agent access to the real utility space for debugging purposes.");
			ExperimentalUserModel e = (ExperimentalUserModel) userModel;
			AbstractUtilitySpace realUSpace = e.getRealUtilitySpace();

			log("The real utility space is: " + realUSpace);
			log("The real utility of the random bid is: "
					+ realUSpace.getUtility(randomBid));
		}
	}

	/**
	 * A simple concession function over time, accepting in the last rounds
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) 
	{
		if (timeline.getType() != Type.Rounds || !hasPreferenceUncertainty())
		{
			log("This agent displays more interesting behavior with a round-based timeline and preference uncertainty; now it simply generates random bids.");
			return new Offer(getPartyId(), generateRandomBid());
		}

		
		// Sample code that accepts offers in the last 3 rounds,
		// or offers that appear in the top 10% of offers in the user model
		if (getLastReceivedAction() instanceof Offer) 
		{
			Bid receivedBid = ((Offer) getLastReceivedAction()).getBid();
			
			// Accept in the last 3 rounds if the received bid is better than the reservation value
			DiscreteTimeline t = (DiscreteTimeline) timeline;
			if (t.getOwnRoundsLeft() < 3 && getUtility(receivedBid) >= utilitySpace.getReservationValue())
				return new Accept(getPartyId(), receivedBid);
			
			List<Bid> bidOrder = userModel.getBidRanking().getBidOrder();

			// If the rank of the received bid is known
			if (bidOrder.contains(receivedBid)) {
				double percentile = (bidOrder.size()
						- bidOrder.indexOf(receivedBid))
						/ (double) bidOrder.size();
				if (percentile < 0.1)
					return new Accept(getPartyId(), receivedBid);
			}
		}
			
		// Return a random, conceding offer above minimumTarget
		Bid randomBid;
		do
		{
			randomBid = generateRandomBid();
			minimumTarget *= 0.999; 
		}
		while (getUtility(randomBid) < minimumTarget);
		return new Offer(getPartyId(), randomBid);
	}

	/**
	 * We override the default estimate of the utility
	 * space by using {@link ClosestKnownBid} defined below.
	 */
	@Override
	public AbstractUtilitySpace estimateUtilitySpace() 
	{
		return new ClosestKnownBid(getDomain());
	}

	@Override
	public String getDescription() {
		return "Example agent with a custom utility space";
	}
	
	/**
	 * Defines a custom UtilitySpace based on the closest known bid to deal with preference uncertainty.
	 */
	private class ClosestKnownBid extends CustomUtilitySpace
	{

		public ClosestKnownBid(Domain dom) {
			super(dom);
		}

		@Override
		public double getUtility(Bid bid) 
		{
			Bid closestRankedBid = getClosestBidRanked(bid);
			return estimateUtilityOfRankedBid(closestRankedBid);
		}
		
		public double estimateUtilityOfRankedBid(Bid b)
		{
			BidRanking bidRanking = getUserModel().getBidRanking();
			Double min = bidRanking.getLowUtility();
			double max = bidRanking.getHighUtility();
			
			int i = bidRanking.indexOf(b);
			
			// index:0 has utility min, index n-1 has utility max
			return min + i * (max - min) / (double) bidRanking.getSize();
		}
		
		/**
		 * Finds the bid in the bid ranking that is most similar to bid given in the argument bid
		 */
		public Bid getClosestBidRanked(Bid bid)
		{
			List<Bid> bidOrder = getUserModel().getBidRanking().getBidOrder();
			Bid closestBid = null;
			double closestDistance = Double.MAX_VALUE;
				
			for (Bid b : bidOrder)
			{
				double d = 1 / (double) b.countEqualValues(bid);
				if (d < closestDistance)
				{
					closestDistance = d;
					closestBid = b;
				}
			}
			return closestBid;
		}
		
	}
	
	private static void log(String s) 
	{
		System.out.println(s);
	}

}
