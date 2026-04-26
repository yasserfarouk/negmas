package genius.gui.progress.session;

import java.util.List;

import javax.swing.DefaultListModel;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.ActionWithBid;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.listener.Listener;
import genius.core.parties.PartyWithUtility;

/**
 * Outcomes model. Listens to {@link NegotiationEvent}s and keeps a
 * {@link DefaultListModel} with the {@link Outcome}s that were reported.
 * 
 * Notice: some panels assume that only 1 item is added to OutcomesModel at a
 * time.
 *
 */
@SuppressWarnings("serial")
public class OutcomesListModel extends DefaultListModel<Outcome> implements Listener<NegotiationEvent> {

	private List<? extends PartyWithUtility> parties;

	public OutcomesListModel(List<? extends PartyWithUtility> parties) {
		this.parties = parties;
	}

	@Override
	public void notifyChange(NegotiationEvent e) {
		Bid bid = null;
		int round = 0, turn = 0;
		boolean isAgreement = false;
		double time = 0;
		AgentID agent = null;
		if (e instanceof MultipartyNegoActionEvent) {
			MultipartyNegoActionEvent event = (MultipartyNegoActionEvent) e;
			if (event.getAction() instanceof ActionWithBid) {
				round = event.getRound();
				turn = event.getTurn();
				bid = ((ActionWithBid) event.getAction()).getBid();
				agent = ((ActionWithBid) event.getAction()).getAgent();
				time = event.getTime();
			}
		} else if (e instanceof SessionEndedNormallyEvent) {
			SessionEndedNormallyEvent event = (SessionEndedNormallyEvent) e;
			bid = event.getAgreement();
			round = ((SessionEndedNormallyEvent) e).getSession().getRoundNumber();
			turn = ((SessionEndedNormallyEvent) e).getSession().getTurnNumber();
			isAgreement = true;
			if (event.getSession().getMostRecentAction() != null) {
				agent = event.getSession().getMostRecentAction().getAgent();
			} else {
				// crash in first round of first agent.
				agent = ((SessionEndedNormallyEvent) e).getParties().get(0).getID();
			}
			time = event.getSession().getTimeline().getTime();
		}
		// FIXME handle other cases?
		if (bid != null) {
			addElement(new Outcome(bid, round, turn, parties, isAgreement, agent, time));
		}
	}

	/**
	 * @return the list of parties in this negotiation.
	 */
	public List<? extends PartyWithUtility> getParties() {
		return parties;
	}

}
