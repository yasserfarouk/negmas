package agents.anac.y2014.AgentYK;


public class PairBidElementDetails {
	private PairBidElement pairBidElement;
	double time;
	
	public PairBidElementDetails(PairBidElement pairBidElement, double time) {
		this.pairBidElement = pairBidElement;
		this.time = time;
	}
	
	public PairBidElement getPairBidElement() {
		return this.pairBidElement;
	}
	
	public double getTime() {
		return this.time;
	}
}
