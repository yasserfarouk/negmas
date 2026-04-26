package agents.nastyagent;

import genius.core.AgentID;
import genius.core.actions.Action;

public class SleepInReceiveMessage extends NastyAgent {

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		try {
			Thread.sleep(2000000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}
