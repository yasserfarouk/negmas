package boa;

import java.util.List;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.OMStrategy;

public class OMStrategy1 extends OMStrategy {

	@Override
	public BidDetails getBid(List<BidDetails> bidsInRange) {
		return null;
	}

	@Override
	public boolean canUpdateOM() {
		return false;
	}

	@Override
	public String getName() {
		return "OMStrategy 1";
	}

}
