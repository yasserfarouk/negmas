package genius.core.events;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.logging.XmlLogger;
import genius.core.misc.ExceptionTool;
import genius.core.session.Participant;
import genius.core.session.Session;
import genius.core.session.SessionConfiguration;
import genius.gui.progress.DataKey;
import genius.gui.progress.DataKeyTableModel;

/**
 * Indicates that a session failed in abnormal way, typically due to an
 * exception, timeout etc.
 * 
 * @author W.Pasman 15jul15
 *
 */
public class SessionFailedEvent implements SessionEndedEvent {

	private final BrokenPartyException exception;

	/**
	 * @param e
	 *            the exception that caused the problem.
	 */
	public SessionFailedEvent(BrokenPartyException e) {
		if (e == null)
			throw new NullPointerException("SessionFailed without cause");
		exception = e;
	}

	public BrokenPartyException getException() {
		return exception;
	}

	public String toString() {
		return "Session " + exception.getSession() + " failed: " + new ExceptionTool(exception).getFullMessage() + ":"
				+ exception.getConfiguration().getParticipantNames();
	}

	/**
	 * Convert the agreement into a hashmap of < {@link DataKey}, {@link Object}
	 * > pairs. Object will usually be a {@link String}, {@link Number} or
	 * {@link List}. This data can be inserted directly into a
	 * {@link DataKeyTableModel}. This is similar to code in {@link XmlLogger}.
	 * 
	 * @return {@link Map} of agreement evaluations.
	 */
	public Map<DataKey, Object> getValues() {
		Map<DataKey, Object> values = new HashMap<DataKey, Object>();
		Session session = exception.getSession();
		SessionConfiguration configuration = exception.getConfiguration();

		try {
			values.put(DataKey.EXCEPTION, toString());
			values.put(DataKey.ROUND, "" + (session.getRoundNumber() + 1));
			values.put(DataKey.DEADLINE, session.getDeadlines().valueString());

			// discounted and agreement
			boolean isDiscounted = false;
			// for (Participant party : configuration.getParties()) {
			values.put(DataKey.IS_AGREEMENT, "No");

			// number of agreeing parties
			values.put(DataKey.NUM_AGREE, 0);

			// utils=reservation values.
			List<Double> utils = new ArrayList<>();
			List<String> agts = new ArrayList<String>();
			List<String> files = new ArrayList<String>();
			for (Participant party : configuration.getParties()) {
				agts.add(party.getStrategy().getUniqueName());
				Double utility = 0d;
				try {
					utility = party.getProfile().create().getReservationValue();
				} catch (Exception e) {
					System.out.println("Failed to read profile of " + party + ". using 0");
				}

				utils.add(utility);
				files.add(party.getProfile().getName());

			}

			values.put(DataKey.AGENTS, agts);
			values.put(DataKey.FILES, files);

			values.put(DataKey.UTILS, utils);
			values.put(DataKey.DISCOUNTED_UTILS, utils);

		} catch (Exception e) {
			e.printStackTrace();
		}
		return values;

	}

}
