package agents.anac.y2010.AgentSmith;

import genius.core.AgentID;
import genius.core.actions.Action;
import genius.core.utility.AdditiveUtilitySpace;

/*
 * An abstract class for the bidstrategy classes
 * Params: the bidhistory for managing what has been offered before, the standard utilityspace,
 *  the preferenceprofilemanager for modelling the opponents' preferences, and the default agentid
 */
public abstract class ABidStrategy {
	protected BidHistory fBidHistory;
	protected AdditiveUtilitySpace fUtilitySpace;
	protected PreferenceProfileManager fPreferenceProfile;
	protected AgentID fAgentID;
	
	/*
	 * Constructor 
	 */
	public ABidStrategy(BidHistory pHist, AdditiveUtilitySpace pUtilitySpace, PreferenceProfileManager pPreferenceProfile, AgentID pId){
		this.fBidHistory = pHist;
		this.fUtilitySpace = pUtilitySpace;
		this.fPreferenceProfile = pPreferenceProfile;
		this.fAgentID = pId;
	}
	
	
	/**
	 * Selects the next action the agent should perform.
	 * @return the next action based on the bidhistory and preference profile.
	 */
	public Action getNextAction(double startTime){
		Action lAction = null;
		return lAction;
	}
}
