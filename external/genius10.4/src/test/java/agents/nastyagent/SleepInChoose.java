package agents.nastyagent;

import java.util.List;

import genius.core.actions.Action;

public class SleepInChoose extends NastyAgent {

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		try {
			Thread.sleep(2000000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return null;
	}
}
