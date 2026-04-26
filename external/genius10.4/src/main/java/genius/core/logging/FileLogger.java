package genius.core.logging;

import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;

import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.listener.Listener;
import genius.core.parties.NegotiationPartyInternal;

/**
 * Creates a file logger which will log the nego events to a csv file
 */
public class FileLogger implements Listener<NegotiationEvent>, Closeable {
	// The internal print stream used for file writing
	PrintStream ps;

	/**
	 * 
	 * @param fileName
	 *            the log file without the .csv extension. If parent
	 *            directory(s) of file don't exist, they are created
	 * @throws FileNotFoundException
	 */
	public FileLogger(String fileName) throws FileNotFoundException {
		new File(fileName).getParentFile().mkdirs();
		ps = new PrintStream(fileName + ".csv");
	}

	@Override
	public void close() throws IOException {
		ps.close();
	}

	@Override
	public void notifyChange(NegotiationEvent e) {
		if (e instanceof MultipartyNegoActionEvent) {
			MultipartyNegoActionEvent ea = (MultipartyNegoActionEvent) e;
			Action action = ea.getAction();
			if (action == null)
				return; // error?
			ps.println(
					ea.getRound() + "," + ea.getTurn() + "," + ea.getTime() + "," + action.getAgent() + "," + action);
		} else if (e instanceof SessionEndedNormallyEvent) {
			SessionEndedNormallyEvent ee = (SessionEndedNormallyEvent) e;
			Bid bid = ee.getAgreement();
			if (bid == null) {
				ps.print("ended-no-agreement");
			} else {
				ps.print("agreement," + bid);
				for (NegotiationPartyInternal party : ((SessionEndedNormallyEvent) e).getParties()) {
					ps.print("," + party.getUtility(bid));
				}
			}
			ps.println();
		}

	}
}
