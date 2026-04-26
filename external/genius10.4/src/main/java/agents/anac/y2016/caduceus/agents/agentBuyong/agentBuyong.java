package agents.anac.y2016.caduceus.agents.agentBuyong;

import java.util.List;

import genius.core.AgentID;
import genius.core.actions.Action;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Created by tdgunes on 30/03/16.
 */
public class agentBuyong extends AbstractNegotiationParty {

	agents.anac.y2015.agentBuyogV2.AgentBuyogMain realAgentBuyong = new agents.anac.y2015.agentBuyogV2.AgentBuyogMain();

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		realAgentBuyong.init(info);

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> list) {
		return realAgentBuyong.chooseAction(list);

	}

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		realAgentBuyong.receiveMessage(sender, arguments);
	}

	@Override
	public String getDescription() {
		return "anac y2016 agentBuyong";
	}
}
