package negotiator.boaframework.offeringstrategy.other;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.OfferingStrategy;

/**
 * This class implements the Simple Agent a.k.a. Zero Intelligence, Random
 * Walker offering strategy. This will choose a bid at random to offer the
 * opponent.
 * 
 * This strategy has no straightforward extension of using opponent models.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class Random_Offering extends OfferingStrategy {

	/**
	 * Empty constructor used for reflection. Note this constructor assumes that
	 * init is called next.
	 */
	public Random_Offering() {
	}

	@Override
	public BidDetails determineNextBid() {
		Bid bid = negotiationSession.getUtilitySpace().getDomain().getRandomBid(null);
		try {
			nextBid = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid),
					negotiationSession.getTime());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return nextBid;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public String getName() {
		return "Other - Random Offering";
	}
}