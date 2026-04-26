package agents.anac.y2016.caduceus.agents.Atlas3;

import java.util.List;

import genius.core.AgentID;
import genius.core.actions.Action;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Created by tdgunes on 30/03/16.
 */
public class Atlas3 extends AbstractNegotiationParty {

	agents.anac.y2015.Atlas3.Atlas3 realAtlas = new agents.anac.y2015.Atlas3.Atlas3();

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		realAtlas.init(info);

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> list) {
		return realAtlas.chooseAction(list);

	}

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		realAtlas.receiveMessage(sender, arguments);
	}

	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		return "anac 2016 Atlas3";
	}
}
