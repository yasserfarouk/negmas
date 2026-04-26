package agents.anac.y2016.caduceus.agents.Caduceus;

import java.util.HashMap;

import genius.core.AgentID;

/**
 * Created by burakatalay on 20/03/16.
 */
public class OpponentProfiles {

	private HashMap<AgentID, Opponent> opponentProfiles = new HashMap<>();

	public HashMap<AgentID, Opponent> getOpponentProfiles() {
		return opponentProfiles;
	}

	public void setOpponentProfile(AgentID sender, Opponent opponentProfile) {
		this.opponentProfiles.put(sender, opponentProfile);
	}
}
