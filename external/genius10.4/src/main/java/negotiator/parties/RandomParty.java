package negotiator.parties;

/**
 * Copies of the random party , needed for assignment to have 2 random agents in
 * tournament.
 *
 */
public class RandomParty extends RandomCounterOfferNegotiationParty 
{
	@Override
	public String getDescription() 
	{
		return super.getDescription() + " (copy)";
	}
}
