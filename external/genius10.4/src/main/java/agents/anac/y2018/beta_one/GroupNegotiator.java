package agents.anac.y2018.beta_one;

import java.util.List;

import java.util.HashMap;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfoList;

public abstract class GroupNegotiator extends AbstractNegotiationParty
{
	protected double negotiationTime;
	protected Offer lastReceivedOffer;
	protected SortedOutcomeSpace outcomeSpace;
	protected HashMap<AgentID, Range> acceptableRanges;

	protected abstract void initialize();

	protected abstract void initializeHistory(StandardInfoList infoList);

	@Override
	public void init(NegotiationInfo info)
	{
		super.init(info);

		outcomeSpace = new SortedOutcomeSpace(getUtilitySpace());
		acceptableRanges = new HashMap<AgentID, Range>();

		initialize();

		if (!utilitySpace.isDiscounted() && getData().getPersistentDataType() == PersistentDataType.STANDARD)
		{
			initializeHistory((StandardInfoList) getData().get());
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action action)
	{
		super.receiveMessage(sender, action);

		if (action instanceof Offer)
		{
			lastReceivedOffer = (Offer) action;
			double utility = utilitySpace.getUtility(lastReceivedOffer.getBid());
			receiveOffer(lastReceivedOffer, utility);
		}
	}

	public void receiveOffer(Offer receivedOffer, double utility)
	{
	}

	public Action chooseAction(List<Class<? extends Action>> possibleActions)
	{
		if (lastReceivedOffer == null)
			return new Offer(getPartyId(), outcomeSpace.getMaxBidPossible().getBid());

		negotiationTime = timeline.getTime();

		// If received offer is acceptable, accept offer
		AgentID receivedAgent = getLastReceivedAction().getAgent();

		double offerUtility = utilitySpace.getUtility(lastReceivedOffer.getBid());

		if (isAcceptable(receivedAgent, offerUtility))
			return new Accept(getPartyId(), lastReceivedOffer.getBid());

		// Create and return new counter offer
		Bid newOffer = createOffer(lastReceivedOffer, offerUtility);
		double newUtility = utilitySpace.getUtility(newOffer);

		if (utilitySpace.isDiscounted())
			newUtility = utilitySpace.discount(newUtility, negotiationTime);

		return new Offer(getPartyId(), newOffer);
	}

	public Bid createOffer(Offer receivedOffer, double utility)
	{
		AgentID receivedAgent = receivedOffer.getAgent();
		Range acceptableRange = getAcceptableRange(receivedAgent);

		List<BidDetails> acceptableBids = outcomeSpace.getBidsinRange(acceptableRange);

		return acceptableBids.get(new Random().nextInt(acceptableBids.size())).getBid();
	}

	public boolean isAcceptable(AgentID agentID, double utility)
	{
		if (utility >= getAcceptableRange(agentID).getLowerbound())
			return true;

		return false;
	}

	public Range getAcceptableRange(AgentID agentID)
	{
		if (!acceptableRanges.containsKey(agentID))
			acceptableRanges.put(agentID, new Range(1, 1));

		return acceptableRanges.get(agentID);
	}

	public void setAcceptableRange(AgentID agentID, Range range)
	{
		acceptableRanges.put(agentID, range);
	}

	public void setAcceptableRange(AgentID agentID, double lowerBound, double upperBound)
	{
		setAcceptableRange(agentID, new Range(lowerBound, upperBound));
	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }
}
