package genius.gui.progress.session;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.awt.BorderLayout;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import org.junit.Test;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Deadline;
import genius.core.DeadlineType;
import genius.core.Domain;
import genius.core.actions.Offer;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.parties.PartyWithUtility;
import genius.core.parties.SessionsInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.protocol.StackedAlternatingOffersProtocol;
import genius.core.session.Session;
import genius.core.timeline.ContinuousTimeline;
import genius.core.utility.UtilitySpace;
import genius.core.xml.SimpleElement;

/***
 * Simple test that runs the progress UI with random data for quick testing
 *
 */
public class SessionProgressUiTest {

	private static final int ROUNDS = 100;
	private final static int NPARTIES = 3;
	private static ArrayList<PartyWithUtility> partiesList = new ArrayList<>();
	private static Session session;
	private static List<NegotiationPartyInternal> parties = new ArrayList<NegotiationPartyInternal>();
	private static SessionsInfo info;

	@Test
	public void runSessionInProgressPanel() throws IOException, InterruptedException {
		NegotiationPartyInternal partyInternal = mock(NegotiationPartyInternal.class);
		Domain domain = mock(Domain.class);
		UtilitySpace utilspace = mock(UtilitySpace.class);
		when(utilspace.getDomain()).thenReturn(domain);

		when(partyInternal.getTimeLine()).thenReturn(mock(ContinuousTimeline.class));
		when(partyInternal.getUtilitySpace()).thenReturn(utilspace);
		parties.add(partyInternal);

		info = new SessionsInfo(new StackedAlternatingOffersProtocol(), PersistentDataType.DISABLED, false);

		for (int n = 0; n < NPARTIES; n++) {
			partiesList.add(new myParty(n));
		}
		session = new Session(new Deadline(ROUNDS, DeadlineType.ROUND), info);

		final OutcomesListModel model = new OutcomesListModel(partiesList);
		final ActionDocumentModel actiondocument = new ActionDocumentModel();

		final JFrame gui = new JFrame();
		gui.setLayout(new BorderLayout());
		gui.getContentPane().add(new SessionProgressUI(model, actiondocument, true, false, false), BorderLayout.CENTER);
		gui.pack();
		gui.setVisible(true);

		Thread thread = new Thread(new Runnable() {
			@Override
			public void run() {
				insertRandomData(model, actiondocument);
				insertEnd(model, actiondocument);
			}

		});
		thread.start();
		thread.join();
		gui.setVisible(false);

		// TODO check actual output in various panels?
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		new SessionProgressUiTest().runSessionInProgressPanel();
	}

	/**
	 * Insert random negotiation events in the models
	 * 
	 * @param model
	 * @param actiondocument
	 */
	private static void insertRandomData(OutcomesListModel model, ActionDocumentModel actiondocument) {
		for (int round = 0; round < ROUNDS; round++) {
			for (int turn = 0; turn < NPARTIES; turn++) {
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				model.addElement(new Outcome(null, round, turn, model.getParties(), Math.random() > 0.9,
						new AgentID("party1"), (double) round / ROUNDS));
				actiondocument.notifyChange(new MultipartyNegoActionEvent(
						new Offer(model.getParties().get(turn).getID(), new Bid((Domain) null)), round, turn,
						(double) round / ROUNDS, null, null));
			}
		}

	}

	/**
	 * Insert nego-finished in the models
	 * 
	 * @param model
	 * @param actiondocument
	 */

	private static void insertEnd(OutcomesListModel model, ActionDocumentModel actiondocument) {

		actiondocument.notifyChange(new SessionEndedNormallyEvent(session, new Bid((Domain) null), parties));
	}

}

/**
 * Support class for the main() example code.
 */
class myParty implements PartyWithUtility {

	private AgentID id;
	private UtilitySpace utilspace;

	public myParty(int n) {
		this.id = new AgentID("PartyNr" + n);
		utilspace = new myUtilSpace();
	}

	@Override
	public AgentID getID() {
		return id;
	}

	@Override
	public UtilitySpace getUtilitySpace() {
		return utilspace;
	}
}

class myUtilSpace implements UtilitySpace {

	@Override
	public Domain getDomain() {
		return null;
	}

	@Override
	public double getUtility(Bid bid) {
		return Math.random();
	}

	@Override
	public Double discount(double util, double time) {
		return 0.8 * util;
	}

	@Override
	public UtilitySpace copy() {
		return null;
	}

	@Override
	public String isComplete() {
		return null;
	}

	@Override
	public SimpleElement toXML() throws IOException {
		return null;
	}

	@Override
	public Double getReservationValue() {
		return null;
	}

	@Override
	public String getName() {
		return null;
	}
}
