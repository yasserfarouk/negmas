package genius.gui.progress;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;

import genius.core.logging.CsvLogger;
import genius.core.session.MultipartyNegoEventLoggerData;
import genius.core.session.TournamentManager;
import genius.gui.negosession.MultiPartyDataModel;
import genius.gui.tournament.MultiTournamentPanel;
import joptsimple.internal.Strings;

/**
 * Logger for MultiPartyNegotiationEvents. Currently only for hook into the
 * {@link TournamentManager} but may be generalizable and eg used in
 * {@link MultiTournamentPanel}. The logger simply listens to changes in the
 * tableModel.
 * 
 * 
 * @author W.Pasman 18jun15
 *
 */
public class MultipartyNegoEventLogger implements TableModelListener {

	private MultipartyNegoEventLoggerData data = new MultipartyNegoEventLoggerData();

	MultiPartyDataModel model;

	/**
	 * 
	 * @param name
	 *            filename but without .csv extension.
	 * @param numAgents
	 * @param m
	 * @throws IOException
	 *             if log file can't be created
	 */
	public MultipartyNegoEventLogger(String name, int numAgents, MultiPartyDataModel m) throws IOException {
		model = m;
		data.logger = new CsvLogger(name + ".csv");
		logHeader();

	}

	/**
	 * write the header to the log file.
	 */
	private void logHeader() {
		List<String> headers = new ArrayList<String>();
		for (int col = 0; col < model.getColumnCount(); col++) {
			headers.add(model.getColumnName(col));
		}
		data.logger.logLine(Strings.join(headers, ";"));

	}

	/**
	 * Any insert in the model is caught here, to be logged. All values in the
	 * new row are added, converted to string
	 */
	@Override
	public void tableChanged(TableModelEvent evt) {
		if (evt.getType() == TableModelEvent.INSERT) {
			int row = evt.getFirstRow();

			List<String> elements = new ArrayList<String>();
			for (int col = 0; col < model.getColumnCount(); col++) {
				Object value = model.getValueAt(row, col);
				elements.add(value == null ? "" : value.toString());
			}

			data.logger.logLine(Strings.join(elements, ";"));
		}
	}

	public void close() {
		if (data.logger != null) {
			try {
				data.logger.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

}