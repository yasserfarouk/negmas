package genius.gui.progress;

import java.awt.BorderLayout;
import java.awt.Panel;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import genius.core.events.NegotiationEvent;
import genius.core.events.TournamentEndedEvent;
import genius.core.events.TournamentSessionStartedEvent;
import genius.core.listener.Listener;

@SuppressWarnings("serial")
public class MultiPartyTournamentProgressUI extends Panel implements Listener<NegotiationEvent> {

	Progress progress = new Progress();

	DataKeyTableModel model;

	/**
	 * @param m
	 *            the {@link DataKeyTableModel} that holds the table to be
	 *            logged.
	 */
	public MultiPartyTournamentProgressUI(DataKeyTableModel m) {
		model = m;
		setLayout(new BorderLayout());

		add(progress, BorderLayout.NORTH);

		// table must be inside scrollpane, otherwise the headers do not show.
		JTable resultsTable = new JTable(model);
		resultsTable.setShowGrid(true); // no effect?
		resultsTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		JScrollPane tableScrollPane = new JScrollPane(resultsTable);

		add(tableScrollPane, BorderLayout.CENTER);
	}

	/***************
	 * implements Listener
	 *********************/

	// we only listen to update the progress bar. The model updates itself.
	@Override
	public void notifyChange(NegotiationEvent e) {
		if (e instanceof TournamentEndedEvent) {
			progress.finish();
		} else if (e instanceof TournamentSessionStartedEvent) {
			TournamentSessionStartedEvent e1 = (TournamentSessionStartedEvent) e;
			progress.update(e1.getCurrentSession(), e1.getTotalSessions());
		}

	}
}

/**
 * progress panel, shows progress bar and text n/N
 *
 */
@SuppressWarnings("serial")
class Progress extends JPanel {
	private final int SCALE = 1000000;
	private JProgressBar progressbar = new JProgressBar(0, SCALE);
	private JLabel label = new JLabel("starting tournament");

	public Progress() {
		setLayout(new BorderLayout());
		add(label, BorderLayout.EAST);
		add(progressbar, BorderLayout.CENTER);
	}

	/**
	 * Shows progress bar when n of total have been started. We are still
	 * working on the nth, even if it equals the total. Therefore the progress
	 * bar never will go exactly to 100%
	 * 
	 * @param n
	 * @param total
	 */
	public void update(int n, int total) {
		progressbar.setValue(Math.min(SCALE - 1, (int) (SCALE * n / (total + 1))));
		label.setText("" + n + "/" + total);
	}

	/**
	 * Set progress bar to 100% of total.
	 * 
	 * @param total
	 */
	public void finish() {
		progressbar.setValue(SCALE);
	}

}