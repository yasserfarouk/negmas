package agents.nastyagent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.list.Tuple;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

/**
 * Checks if data is retained in persistent storage properly. This uses a hack.
 *
 * 
 */
public class CheckStoredData extends NastyAgent {
	private static final String DATA = "data";
	static boolean runBefore = false;
	static List<ActionWithBid> previousActions = new ArrayList<ActionWithBid>();

	public void init(NegotiationInfo info) {
		super.init(info);
		if (runBefore) {
			checkStore((StandardInfoList) data.get());
		}
	}

	private void checkStore(StandardInfoList standardInfoList) {
		if (standardInfoList.size() != 1) {
			throw new IllegalStateException("Expected 1 info but found " + standardInfoList.size());
		}
		StandardInfo info = standardInfoList.get(0);
		if (info.getStartingAgent() == previousActions.get(0).getAgent().toString()) {
			throw new IllegalStateException("the starting agent is not reported properly");
		}

		List<Tuple<String, Double>> utilities = info.getUtilities();
		if (utilities.size() != previousActions.size()) {
			throw new IllegalStateException("The number of actions is not the expected " + previousActions.size());
		}

		for (int n = 0; n < previousActions.size(); n++) {
			if (!utilities.get(n).get1().startsWith(previousActions.get(n).getAgent().toString())) {
				throw new IllegalStateException("agent at utility " + n + " differs");
			}
			if (utilities.get(n).get2() != utilitySpace.getUtility(previousActions.get(n).getBid())) {
				throw new IllegalStateException("utility " + n + " differs");
			}
		}

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		Action action = super.chooseAction(possibleActions);
		if (!runBefore && action instanceof ActionWithBid) {
			previousActions.add((ActionWithBid) action);
		}
		return action;
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		if (!runBefore && action instanceof ActionWithBid) {
			previousActions.add((ActionWithBid) action);
		}
	}

	@Override
	public HashMap<String, String> negotiationEnded(Bid acceptedBid) {
		// we actually got here. Report it.
		super.negotiationEnded(acceptedBid);
		runBefore = true;

		return null;
	}

}
