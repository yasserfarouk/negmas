package genius.core.logging;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.stream.XMLStreamException;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Offer;
import genius.core.events.AgentLogEvent;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.listener.Listener;
import genius.core.misc.ExceptionTool;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.session.Participant;
import genius.core.session.Session;
import genius.core.session.SessionConfiguration;
import genius.core.utility.UtilitySpace;
import genius.core.xml.Key;
import genius.core.xml.XmlWriteStream;

/**
 * Creates a logger which will log {@link NegotiationEvent}s to a XML file. Logs
 * the {@link SessionEndedNormallyEvent}.
 */
public class XmlLogger implements Listener<NegotiationEvent>, Closeable {

	private XmlWriteStream stream;
	/**
	 * map<key,value> where keys are the agent names. The values are the logs
	 * returned by the agent through an {@link AgentLogEvent}.
	 */
	protected Map<String, Map<Object, Object>> agentLogs = new HashMap<>();
	private int nrOffers = 0;
	/**
	 * The agent that did the first action. Null until a first action was done
	 * in the current session or if there is no current session.
	 */
	private AgentID startingAgent = null;

	/**
	 * @param out
	 *            {@link OutputStream} to write the log to. If this is a file,
	 *            we recommend to use the extension ".xml". This logger becomes
	 *            owner of this outputstream and will close it eventually.
	 * @param topLabel
	 *            the top level label to use in the output file
	 * @throws FileNotFoundException
	 * @throws XMLStreamException
	 */
	public XmlLogger(OutputStream out, String topLabel) throws FileNotFoundException, XMLStreamException {
		stream = new XmlWriteStream(out, topLabel);
	}

	@Override
	public void close() throws IOException {
		try {
			stream.flush();
		} catch (XMLStreamException e) {
			e.printStackTrace();
		}
		stream.close();
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	@Override
	public void notifyChange(NegotiationEvent e) {
		try {
			if (e instanceof MultipartyNegoActionEvent) {
				if (startingAgent == null) {
					startingAgent = ((MultipartyNegoActionEvent) e).getAction().getAgent();
				}
				if (((MultipartyNegoActionEvent) e).getAction() instanceof Offer) {
					nrOffers++;
				}
			} else if (e instanceof AgentLogEvent) {
				// Map<String,String> to Map<Object,Object>...
				agentLogs.put(((AgentLogEvent) e).getAgent(), (Map<Object, Object>) (Map) ((AgentLogEvent) e).getLog());
			} else if (e instanceof SessionEndedNormallyEvent) {
				stream.write("NegotiationOutcome", getOutcome((SessionEndedNormallyEvent) e));
				reset();
			} else if (e instanceof SessionFailedEvent) {
				stream.write("NegotiationOutcome", getOutcome((SessionFailedEvent) e));
				reset();
			}
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * Flush streams, reset counters etc.
	 * 
	 * @throws XMLStreamException
	 */
	private void reset() {
		try {
			stream.flush();
		} catch (XMLStreamException e) {
			e.printStackTrace();
		}
		// log done, reset the per-session trackers
		agentLogs = new HashMap<>();
		nrOffers = 0;
		startingAgent = null;

	}

	/**
	 * Mostly duplicate from the other getOutcome but for failed sessions.
	 */
	private Map<Object, Object> getOutcome(SessionFailedEvent e) {
		Map<Object, Object> outcome = new HashMap<>();

		Session session = e.getException().getSession();
		outcome.put("exception", new ExceptionTool(e.getException()).getFullMessage());
		outcome.put("currentTime", new Date().toString());
		outcome.put("bids", nrOffers);

		outcome.put("lastAction", session.getMostRecentAction());
		outcome.put("deadline", session.getDeadlines().valueString());
		outcome.put("runtime", session.getRuntimeInSeconds());
		outcome.putAll(getPartiesReservationPoints(e.getException().getConfiguration()));

		return outcome;
	}

	/**
	 * 
	 * @param configuration
	 *            the {@link SessionConfiguration}.
	 * @return list of reservation values for all participants in the
	 *         configuration.
	 */
	private Map<Object, Object> getPartiesReservationPoints(SessionConfiguration configuration) {
		Map<Object, Object> outcome = new HashMap<>();

		for (Participant party : configuration.getParties()) {
			outcome.put(new Key("resultsOfAgent"), getReservationValue(party));
		}

		return outcome;

	}

	/**
	 * Get info based only on participant info. This is tricky as we need to
	 * consider many cases, especially because we get here because something is
	 * not quite ok in this info.
	 * 
	 * @param party
	 *            the {@link Participant}
	 * @return object containing as much info as we can get safely from this.
	 */
	private Object getReservationValue(Participant party) {
		Map<Object, Object> outcome = new HashMap<>();
		outcome.put("agent", party.getStrategy().getUniqueName());
		outcome.put("agentClass", party.getStrategy().getClassDescriptor());
		outcome.put("utilspace", party.getProfile().getURL().getFile());
		Double finalUtility = 0d;
		Double discount = 0d;
		Double discountedUtility = finalUtility;
		try {
			UtilitySpace utilspace = party.getProfile().create();
			finalUtility = discountedUtility = utilspace.getReservationValue();
			discount = utilspace.discount(1.0, 1.0);
		} catch (Exception e) {
			System.out.println("Failed to read profile of " + party + ". using 0");
		}
		outcome.put("discount", discount);
		outcome.put("finalUtility", finalUtility);
		outcome.put("discountedUtility", discountedUtility);

		return outcome;
	}

	/**
	 * @param e
	 *            the {@link SessionEndedNormallyEvent}
	 * @return the complete session outcome, including all agent outcomes
	 */
	private Map<Object, Object> getOutcome(SessionEndedNormallyEvent e) {
		Map<Object, Object> outcome = new HashMap<>();

		Session session = e.getSession();
		outcome.put("currentTime", new Date().toString());
		outcome.put("startingAgent", startingAgent == null ? "-" : startingAgent.toString());
		outcome.put("bids", nrOffers);

		outcome.put("lastAction", session.getMostRecentAction());
		outcome.put("deadline", session.getDeadlines().valueString());
		outcome.put("runtime", session.getRuntimeInSeconds());
		outcome.put("domain", e.getParties().get(0).getUtilitySpace().getDomain().getName());
		outcome.put("finalOutcome", e.getAgreement() == null ? "-" : e.getAgreement());

		if (e.getAgreement() != null) {
			outcome.put("timeOfAgreement", session.getTimeline().getTime());
		}

		outcome.putAll(getAgentResults(e.getParties(), e.getAgreement()));
		return outcome;
	}

	/**
	 * 
	 * @param parties
	 *            the parties in the negotiation
	 * @param bid
	 *            the accepted bid, or null
	 * @return a Map containing all party results as key-value pairs
	 */
	private Map<Object, Object> getAgentResults(List<NegotiationPartyInternal> parties, Bid bid) {
		Map<Object, Object> outcome = new HashMap<>();

		for (NegotiationPartyInternal party : parties) {
			outcome.put(new Key("resultsOfAgent"), partyResults(party, bid));
		}

		return outcome;
	}

	/**
	 * Collect the results of a party in a map. This map will also contain the
	 * logs as done by the agent.
	 * 
	 * @param party
	 *            the party to collect the results for
	 * @param bid
	 *            the accepted bid, or null
	 * @return a map containing the results
	 */
	private Map<Object, Object> partyResults(NegotiationPartyInternal party, Bid bid) {
		Map<Object, Object> outcome = new HashMap<>();
		if (agentLogs.containsKey(party.getID())) {
			outcome.putAll(agentLogs.get(party.getID()));
		}
		outcome.put("agent", party.getID());
		outcome.put("agentClass", party.getParty().getClass().getName());
		outcome.put("agentDesc", party.getParty().getDescription());
		outcome.put("utilspace", party.getUtilitySpace().getName());
		outcome.put("discount", party.getUtilitySpace().discount(1.0, 1.0));
		outcome.put("totalUserBother", (party.getUser()!=null) ? party.getUser().getTotalBother() : 0.0);
		outcome.put("finalUtility", party.getUtility(bid));
		outcome.put("discountedUtility", party.getUtilityWithDiscount(bid));
		outcome.put("userUtility", (party.getUser()!=null) ? party.getUtilityWithDiscount(bid)-party.getUser().getTotalBother():party.getUtilityWithDiscount(bid));
		
		return outcome;
	}

}
