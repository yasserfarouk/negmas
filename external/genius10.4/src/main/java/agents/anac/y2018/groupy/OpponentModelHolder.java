package agents.anac.y2018.groupy;
import java.util.HashMap;

import genius.core.AgentID;

public class OpponentModelHolder {
	private HashMap<AgentID,OpponentModel> agentModels;
	public OpponentModelHolder() {
		agentModels=new HashMap<AgentID,OpponentModel>();
			
		
	}
	public HashMap<AgentID, OpponentModel> getAgentModels() {
		return agentModels;
	}
	
	
	
}
