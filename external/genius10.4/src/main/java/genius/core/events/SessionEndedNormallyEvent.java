package genius.core.events;


import static java.lang.String.format;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.analysis.MultilateralAnalysis;
import genius.core.logging.CsvLogger;
import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.session.Session;
import genius.core.utility.AbstractUtilitySpace;
import genius.gui.progress.DataKey;
import genius.gui.progress.DataKeyTableModel;

/**
 * Abstract superclass indicating end of a session. All events happening after
 * this event are for another session. This is not called if there is a
 * {@link SessionFailedEvent}.
 *
 */
public class SessionEndedNormallyEvent implements SessionEndedEvent {
	private Session session;
	private Bid agreement;
	private List<NegotiationPartyInternal> parties;
	private double runTime;
	private MultilateralAnalysis analysis;
	private ArrayList<NegotiationParty> ps;

	/**
	 * @param session
	 *            the session that ended
	 * @param agreement
	 *            the bid that was agreed on at the end, or null if no
	 *            agreement.
	 * @param parties
	 *            list of the involved {@link NegotiationPartyInternal} , in
	 *            correct order
	 */
	public SessionEndedNormallyEvent(Session session, Bid agreement, List<NegotiationPartyInternal> parties) {
		this.session = session;
		this.agreement = agreement;
		this.parties = parties;
		this.runTime = session.getRuntimeInSeconds();
		ps = new ArrayList<NegotiationParty>();
		for (NegotiationPartyInternal p : parties) {
			ps.add(p.getParty());
		}

		analysis = new MultilateralAnalysis(parties, session.getInfo().getProtocol().getCurrentAgreement(session, ps),
				session.getTimeline().getTime());

	}

	public Session getSession() {
		return session;
	}

	/**
	 * 
	 * @return final agreement bid, or null if no agreement was reached
	 */
	public Bid getAgreement() {
		return agreement;
	}

	public List<NegotiationPartyInternal> getParties() {
		return parties;
	}

	/**
	 * Convert the agreement into a hashmap of < {@link DataKey}, {@link Object}
	 * > pairs. Object will usually be a {@link String}, {@link Number} or
	 * {@link List}. This data can be inserted directly into a
	 * {@link DataKeyTableModel}.
	 * 
	 * @return {@link Map} of agreement evaluations.
	 */
	public Map<DataKey, Object> getValues() {
		Map<DataKey, Object> values = new HashMap<DataKey, Object>();

		try {
			Bid agreement = session.getInfo().getProtocol().getCurrentAgreement(session, ps);
			values.put(DataKey.RUNTIME, format("%.3f", runTime));
			values.put(DataKey.ROUND, "" + (session.getRoundNumber() + 1));

			// deadline
			values.put(DataKey.DEADLINE, session.getDeadlines().valueString());

			// discounted and agreement
			boolean isDiscounted = false;
			for (NegotiationPartyInternal party : parties)
				isDiscounted |= (party.getUtilitySpace().discount(1, 1) != 1);
			values.put(DataKey.IS_AGREEMENT, agreement == null ? "No" : "Yes");
			values.put(DataKey.IS_DISCOUNT, isDiscounted ? "Yes" : "No");

			// number of agreeing parties
			values.put(DataKey.NUM_AGREE, "" + session.getInfo().getProtocol().getNumberOfAgreeingParties(session, ps));

			// disc. and undisc. utils;
			List<Double> utils = CsvLogger.getUtils(parties, agreement, false);
			List<Double> discountedUtils = CsvLogger.getUtils(parties, agreement, true);
			List<Double> perceivedUtils = CsvLogger.getPerceivedUtils(parties, agreement, true);
			
			// user bothers;
			List<Double> userBothers = CsvLogger.getUserBothers(parties);
			List<Double> userUtils = CsvLogger.getUserUtilities(parties, agreement, true);
			
			// min and max discounted utility
			values.put(DataKey.MINUTIL, format("%.5f", Collections.min(discountedUtils)));
			values.put(DataKey.MAXUTIL, format("%.5f", Collections.max(discountedUtils)));

			// analysis (distances, social welfare, etc)
			values.put(DataKey.DIST_PARETO, format("%.5f", analysis.getDistanceToPareto()));
			values.put(DataKey.DIST_NASH, format("%.5f", analysis.getDistanceToNash()));
			values.put(DataKey.SOCIAL_WELFARE, format("%.5f", analysis.getSocialWelfare()));

			// enumerate agents names, utils, protocols
			List<String> agts = new ArrayList<String>();

			String agentstr = "";
			for (NegotiationPartyInternal a : parties) {
				agts.add(a.getID().toString());
			}
			values.put(DataKey.AGENTS, agts);
			values.put(DataKey.UTILS, utils);
			values.put(DataKey.DISCOUNTED_UTILS, discountedUtils);
			values.put(DataKey.PERCEIVED_UTILS, perceivedUtils);
			values.put(DataKey.USER_BOTHERS, userBothers);
			values.put(DataKey.USER_UTILS, userUtils);
			
			List<String> files = new ArrayList<String>();
			for (NegotiationPartyInternal agent : parties) {
				String name = "-";
				if (agent.getUtilitySpace() instanceof AbstractUtilitySpace) {
					name = new File(((AbstractUtilitySpace) agent.getUtilitySpace()).getFileName()).getName();
				}
				files.add(name);
			}
			values.put(DataKey.FILES, files);

		} catch (Exception e) {
			values.put(DataKey.EXCEPTION, e.toString());
		}
		return values;

	}

	public MultilateralAnalysis getAnalysis() {
		return analysis;
	}

}
