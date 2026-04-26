package agents.anac.y2017.tucagent;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;

public class OpponentModel {
	public OpponentModel() {
	}

	private boolean isCrashed = false;
	protected Domain fDomain;
	public java.util.ArrayList<Bid> fBiddingHistory;

	public Domain getDomain() {
		return fDomain;
	}

	Double minUtility = null;
	Double maxUtility = null;

	public double getExpectedUtility(Bid pBid) throws Exception {
		return -1.0D;
	}

	public double getNormalizedUtility(Bid pBid) throws Exception {
		double u = getExpectedUtility(pBid);
		if ((minUtility == null) || (maxUtility == null))
			findMinMaxUtility();
		double value = (u - minUtility.doubleValue()) / (maxUtility.doubleValue() - minUtility.doubleValue());
		if (Double.isNaN(value)) {
			if (!isCrashed) {
				isCrashed = true;
				System.err.println("Bayesian scalable encountered NaN and therefore crashed");
			}
			return 0.0D;
		}

		return (u - minUtility.doubleValue()) / (maxUtility.doubleValue() - minUtility.doubleValue());
	}

	public void updateBeliefs(Bid pBid) throws Exception {
	}

	public double getExpectedWeight(int pIssueNumber) {
		return 0.0D;
	}

	public boolean haveSeenBefore(Bid pBid) {
		for (Bid tmpBid : fBiddingHistory) {
			if (pBid.equals(tmpBid))
				return true;
		}
		return false;
	}

	protected void findMinMaxUtility() throws Exception {
		BidIterator biditer = new BidIterator(fDomain);
		minUtility = Double.valueOf(1.0D);
		maxUtility = Double.valueOf(0.0D);

		while (biditer.hasNext()) {
			Bid b = biditer.next();
			double u = getExpectedUtility(b);

			if (minUtility.doubleValue() > u)
				minUtility = Double.valueOf(u);
			if (maxUtility.doubleValue() < u)
				maxUtility = Double.valueOf(u);
		}
	}

	public boolean isCrashed() {
		return isCrashed;
	}
}
