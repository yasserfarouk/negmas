package agents.anac.y2014.AgentYK;


public class BidElementDetails {
	private BidElement bidElement;
	double time;
	
	public BidElementDetails(BidElement bidElement, double time) {
		this.bidElement = bidElement;
		this.time = time;
	}
	
	public BidElement getBidElement() {
		return this.bidElement;
	}
	
	public double getTime() {
		return this.time;
	}
}
