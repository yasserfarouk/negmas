package genius.core.misc;

import java.util.List;

import genius.core.events.NegotiationEvent;
import genius.core.listener.Listener;
import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.session.SessionManager;
import agents.rlboa.RLBOA;

public class RLBOAUtils {
	/**
	 * Adds any party that conforms to the @RLBOA protocol as a listener to
	 * the @SessionManager this allows these parties to react to negotiation
	 * events and update their parameters.
	 * 
	 * @param parties
	 *            list of participating negotiation parties
	 * @param manager
	 *            the @SessionManager object
	 */
	@SuppressWarnings({ "unchecked" })
	public static void addReinforcementAgentListeners(
			List<NegotiationPartyInternal> parties, SessionManager manager) {
		for (NegotiationPartyInternal partyInternal : parties) {
			NegotiationParty participant = partyInternal.getParty();

			// Every RLBOA agent implements the Listener interface, so we
			// subscribe it.
			if (participant instanceof RLBOA) {
				Listener<NegotiationEvent> participantAsListener = (Listener<NegotiationEvent>) participant;
				manager.addListener(participantAsListener);
			}
		}
	}
}
