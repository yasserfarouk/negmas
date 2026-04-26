package genius.gui.progress.session;

import javax.swing.JTextArea;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;
import javax.swing.text.PlainDocument;

import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.listener.Listener;
import genius.core.logging.CsvLogger;
import genius.core.session.Session;

/**
 * A document containing all actions that occured on the NegotiationEvent
 * stream. Since this is a {@link Document} you can directly use this in a
 * {@link JTextArea}.
 *
 */
@SuppressWarnings("serial")
public class ActionDocumentModel extends PlainDocument implements Listener<NegotiationEvent> {

	private int currentround = -1;

	public ActionDocumentModel() {
		writeln("Starting negotiation session.\n");
	}

	@Override
	public void notifyChange(NegotiationEvent e) {
		if (e instanceof MultipartyNegoActionEvent) {
			MultipartyNegoActionEvent e1 = (MultipartyNegoActionEvent) e;
			int round = e1.getRound();
			if (round != currentround) {
				currentround = round;
				writeln("Round " + currentround);
			}
			writeln(" Turn " + e1.getTurn() + ":" + e1.getAction().getAgent() + "  " + e1.getAction());
		} else if (e instanceof SessionEndedNormallyEvent) {
			SessionEndedNormallyEvent e1 = (SessionEndedNormallyEvent) e;
			if (e1.getAgreement() == null)
				writeln("No agreement found.");
			else {
				writeln("Found an agreement: " + e1.getAgreement());
			}
			Session session = ((SessionEndedNormallyEvent) e).getSession();
			double runTime = session.getRuntimeInSeconds();
			writeln("Finished negotiation session in " + runTime + " s.");
			try {
				writeln(CsvLogger.logSingleSession(((SessionEndedNormallyEvent) e).getSession(),
						session.getInfo().getProtocol(), e1.getParties(), runTime));
			} catch (Exception e2) {
				e2.printStackTrace();
			}

		}
	}

	private void writeln(String text) {
		try {
			insertString(getLength(), text + "\n", null);
		} catch (BadLocationException e) {
			e.printStackTrace();
		}
	}

}
