package agents.anac.y2017.simpleagent;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfoList;

public class SimpleAgent extends AbstractNegotiationParty {

	private Bid currentBid;
	private Predictor predictor;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		predictor = new Predictor(getUtilitySpace(), timeline);
		super.init(info);
		StandardInfoList history = (StandardInfoList) getData().get();
		if (history != null && !history.isEmpty()) {
			System.out.println(
					"Hist" + history.get(0).getUtilities().get(0).get1());
			predictor.setHistoryAndUpdateThreshold(history);
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		try {
			Action action = predictor.generateAction(validActions, currentBid,
					getPartyId());
			return action;

		} catch (Exception e) {
			e.printStackTrace();
			return new Accept(getPartyId(), currentBid); // Not sure what to put
															// here... Don't
															// know what
			// error I would be hitting.
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}
	
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);

		if (action instanceof Offer) {
			predictor.storeAgentOffer((Offer) action);
			currentBid = ((Offer) action).getBid();
		}
	}
}