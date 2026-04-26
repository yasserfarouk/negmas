package agents.anac.y2017.group3;

import java.util.ArrayList;

import genius.core.Bid;

public class BidData {

	String agentName;
	Bid bid;
	ArrayList<Integer> assumedValues = new ArrayList<Integer>();

	public BidData(String agentName, Bid bid) {
		this.agentName = agentName;
		this.bid = bid;
	}

	@Override
	public String toString() {
		return "BidData [agentName=" + agentName + ", bid=" + bid + ", assumedValues=" + assumedValues + "]";
	}

}