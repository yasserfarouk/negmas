package agents.anac.y2014.AgentYK;

import genius.core.issue.Value;

public class BidElement {
	private int issueNr;
	private Value value;
	
	public BidElement(Integer issueNr, Value value) {
		this.issueNr = issueNr;
		this.value = value;
	}
	
	public int getIssueNumber() {
		return this.issueNr;
	}
	
	public Value getValue() {
		return this.value;
	}
	
	public boolean equals(BidElement other) {
		if( (this.issueNr == other.getIssueNumber() && this.value.toString().equals(other.getValue().toString()))) {
			return true;
		}
		return false;
	}
	
	public String toString() {
		String str = "" + issueNr + "," + value.toString();
		return str;
	}
}
