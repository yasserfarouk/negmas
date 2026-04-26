package agents.anac.y2016.caduceus.agents.kawaii;

import java.util.List;

import genius.core.AgentID;
import genius.core.actions.Action;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Created by tdgunes on 30/03/16.
 */
public class kawaii extends AbstractNegotiationParty {

	agents.anac.y2015.fairy.kawaii realkawaii = new agents.anac.y2015.fairy.kawaii();

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		realkawaii.init(info);

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> list) {
		return realkawaii.chooseAction(list);

	}

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		realkawaii.receiveMessage(sender, arguments);
	}

	@Override
	public String getDescription() {
		return "anac 2016 kawaii";
	}
}
