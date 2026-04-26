package genius.core;

import genius.core.events.ActionEvent;

/**
 * implement this class in order to subscribe with the NegotiationManager to get
 * callback on handleEvent().
 *
 */
public interface NegotiationEventListener {
	/**
	 * IMPORTANT: in handleEvent, do not more than just storing the event and
	 * notifying your interface that a new event has arrived. Doing more than
	 * this will snoop time from the negotiation, which will disturb the
	 * negotiation.
	 * 
	 * @param evt
	 */
	public void handleActionEvent(ActionEvent evt);

}
