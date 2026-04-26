package agents.anac.y2014.AgentYK;

import genius.core.misc.Pair;

public class PairBidElement extends Pair<BidElement, BidElement> {
	private static final long serialVersionUID = 19901213L;
	
	/*
	 * first.getIssueNumber() <= second.getIssueNumber() by force.
	 */
	public PairBidElement(BidElement fst, BidElement snd) {
		super(fst, snd);
		if(fst.getIssueNumber() > snd.getIssueNumber()) {
			super.setFirst(snd);
			super.setSecond(fst);
		}
	}
	
	public boolean equals(PairBidElement other) {
		if( (this.getFirst().equals(other.getFirst()) && this.getSecond().equals(other.getSecond())) ||
				(this.getFirst().equals(other.getSecond()) && this.getSecond().equals(other.getFirst()))) {
			return true;
		}
		return false;
	}
	
	public String toString() {
		String str = "" + this.getFirst().getIssueNumber() + ":" + this.getFirst().getValue().toString() + " , "
						+ this.getSecond().getIssueNumber() + ":" + this.getSecond().getValue().toString();
		return str;
	}
}
