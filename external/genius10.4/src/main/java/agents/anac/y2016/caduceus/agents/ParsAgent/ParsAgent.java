package agents.anac.y2016.caduceus.agents.ParsAgent;

import java.util.List;

import genius.core.AgentID;
import genius.core.actions.Action;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Created by tdgunes on 30/03/16.
 */
public class ParsAgent extends AbstractNegotiationParty {

	agents.anac.y2015.ParsAgent.ParsAgent realParsAgent = new agents.anac.y2015.ParsAgent.ParsAgent();

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		realParsAgent.init(info);

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> list) {
		return realParsAgent.chooseAction(list);

	}

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		realParsAgent.receiveMessage(sender, arguments);
	}

	@Override
	public String getDescription() {
		return "anac 2016 ParsAgent";
	}
}
