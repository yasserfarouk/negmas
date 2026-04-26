package genius.core.boaframework;

import genius.core.Bid;

/**
 * This class is a container which holds the necessary information so that a particular NegotiationOutcome
 * of the multi-acceptance criteria (MAC) can be reconstructed given a full negotiation outcome.
 * 
 * The MAC technique runs multiple negotiation outcomes in parallel during a single negotiation. The negotiation
 * ends when the deadline has been reached, or when all acceptance criteria accepted. Normally, a single match
 * results in a single negotiation outcome. However, in this case there are multiple acceptance criteria each
 * resulting in a subset of the full negotiation trace. This class aids in generating this new outcome.
 * 
 * @author Alex Dirkwager
 */
public class OutcomeTuple {
	/** last bid done by an agent */
	private Bid lastBid;
	/** name of the acceptance criteria */
	private String name;
	/** time of acceptance */
	private double time;
	/** amount of bids made by agent A */
	private int agentASize;	
	/** amount of bids made by agent B */
	private int agentBSize;
	/**What type of log messege (deadline reached, breakoff)**/
	private String logMsgType;
	private String acceptedBy;
	
	public OutcomeTuple(Bid lastBid, String name, double time, int agentASize, int agentBSize, String logMsg, String acceptedBy){
		this.lastBid = lastBid;
		this.name = name;
		this.time = time;
		this.agentASize = agentASize;
		this.agentBSize = agentBSize;
		this.logMsgType = logMsg;
		this.acceptedBy = acceptedBy;
	}

	public String getAcceptedBy() {
		return acceptedBy;
	}

	public void setAcceptedBy(String acceptedBy) {
		this.acceptedBy = acceptedBy;
	}

	public int getAgentASize() {
		return agentASize;
	}

	public void setAgentASize(int agentASize) {
		this.agentASize = agentASize;
	}

	public int getAgentBSize() {
		return agentBSize;
	}

	public void setAgentBSize(int agentBSize) {
		this.agentBSize = agentBSize;
	}

	public Bid getLastBid() {
		return lastBid;
	}

	public void setLastBid(Bid lastBid) {
		this.lastBid = lastBid;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public double getTime() {
		return time;
	}

	public void setTime(double time) {
		this.time = time;
	}
	
	public String getLogMsgType(){
		return logMsgType;
	}
	
	public String toString() {
		return "LastBid: " + lastBid + ", Name of AC: " + name + ", Time of agreement: " + time + 
				" agentASize: " + agentASize + " agentBSize: " + agentBSize;
	}
}
