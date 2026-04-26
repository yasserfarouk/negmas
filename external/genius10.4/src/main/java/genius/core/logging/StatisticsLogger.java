package genius.core.logging;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.OutputStream;
import java.util.ArrayList;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.stream.XMLStreamException;

import genius.core.Bid;
import genius.core.analysis.MultilateralAnalysis;
import genius.core.events.BrokenPartyException;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.listener.Listener;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.session.Participant;
import genius.core.utility.UtilitySpace;

/**
 * Keeps track of tournament and creates statistic information.
 *
 */
public class StatisticsLogger implements Listener<NegotiationEvent>, Closeable {

	/**
	 * key= agent name, Statistic is the statistical info logged for that agent.
	 */
	protected AgentsStatistics agentStats = new AgentsStatistics(new ArrayList<AgentStatistics>());
	private OutputStream outStream;

	/**
	 * @param out
	 *            {@link OutputStream} to write the log to. If this is a file,
	 *            we recommend to use the extension ".xml". This logger becomes
	 *            owner of this outputstream and will close it eventually.
	 */
	public StatisticsLogger(OutputStream out) throws FileNotFoundException, XMLStreamException {
		if (out == null) {
			throw new NullPointerException("out=null");
		}
		this.outStream = out;
	}

	@Override
	public void close() {
		try {
			JAXBContext jc = JAXBContext.newInstance(AgentsStatistics.class);
			Marshaller marshaller = jc.createMarshaller();
			marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
			marshaller.marshal(agentStats, outStream);
		} catch (JAXBException e) {
			e.printStackTrace(); // we can't do much here.
		}
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	@Override
	public void notifyChange(NegotiationEvent e) {
		try {
			if (e instanceof SessionEndedNormallyEvent) {
				SessionEndedNormallyEvent e1 = (SessionEndedNormallyEvent) e;
				MultilateralAnalysis analysis = e1.getAnalysis();
				Bid agreedbid = analysis.getAgreement();
				double nashdist = analysis.getDistanceToNash();
				double welfare = analysis.getSocialWelfare();
				double paretoDist = analysis.getDistanceToPareto();

				for (NegotiationPartyInternal party : e1.getParties()) {
					String name = party.getParty().getClass().getCanonicalName();
					if (agreedbid == null) {
						agentStats = agentStats.withStatistics(name, 0, 0, nashdist, welfare, paretoDist);
					} else {
						agentStats = agentStats.withStatistics(name, party.getUtility(agreedbid),
								party.getUtilityWithDiscount(agreedbid), nashdist, welfare, paretoDist);
					}
				}
			} else if (e instanceof SessionFailedEvent) {
				BrokenPartyException e1 = ((SessionFailedEvent) e).getException();

				for (Participant party : ((SessionFailedEvent) e).getException().getConfiguration().getParties()) {
					Double reservationvalue = 0d;
					try {
						UtilitySpace utilspace = party.getProfile().create();
						reservationvalue = utilspace.getReservationValue();
					} catch (Exception ex) {
						System.out.println("Failed to read profile of " + party + ". using 0");
					}

					agentStats = agentStats.withStatistics(party.getStrategy().getClassDescriptor(), reservationvalue,
							reservationvalue, 1d, 0d, 1d);
				}
			}
			// other events are only giving details we dont need here.

		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}