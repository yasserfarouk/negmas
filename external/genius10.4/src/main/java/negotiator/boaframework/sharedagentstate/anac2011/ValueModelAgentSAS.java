package negotiator.boaframework.sharedagentstate.anac2011;

import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC 2011 ValueModelAgent.
 * The code was taken from the ANAC2011 ValueModelAgent and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class ValueModelAgentSAS extends SharedAgentState {

	private double opponentMaxBidUtil = 0;
	private double opponentUtil = -1;
	private double lowestApprovedInitial = 1;
	private double plannedThreshold = 1;
	private double lowestApproved;
	private boolean skipAcceptDueToCrash;
	
	public ValueModelAgentSAS() {
		NAME = "ValueModelAgent";
	}

	public double getOpponentMaxBidUtil() {
		return opponentMaxBidUtil;
	}
	
	public void setOpponentMaxBidUtil(double opponentMaxBidUtil) {
		this.opponentMaxBidUtil = opponentMaxBidUtil;
	}

	public void setOpponentUtil(double opponentUtil) {
		this.opponentUtil = opponentUtil;
	}

	public void setLowestApprovedInitial(double lowestApproved) {
		lowestApprovedInitial = lowestApproved;
	}
	
	public double getLowestApprovedInitial() {
		return lowestApprovedInitial;
	}

	public void setPlanedThreshold(double plannedThreshold) {
		this.plannedThreshold = plannedThreshold;
	}

	public double getOpponentUtil() {
		return opponentUtil;
	}

	public double getPlannedThreshold() {
		return plannedThreshold;
	}

	public void setLowestApproved(double lowestApproved) {
		this.lowestApproved = lowestApproved;
	}
	
	public double getLowestApproved() {
		return lowestApproved;
	}

	public void triggerSkipAcceptDueToCrash() {
		skipAcceptDueToCrash = true;
	}
	
	public boolean shouldSkipAcceptDueToCrash() {
		if (skipAcceptDueToCrash) {
			skipAcceptDueToCrash = false;
			return true;
		}
		return false;
	}
}