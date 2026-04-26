package boa;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.OfferingStrategy;

public class Offering1 extends OfferingStrategy {

	@Override
	public BidDetails determineOpeningBid() {
		return null;
	}

	@Override
	public BidDetails determineNextBid() {
		return null;
	}

	@Override
	public String getName() {
		return "Offering 1";
	}

}
