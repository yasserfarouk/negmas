package genius.gui.progress.session;

import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTable;
import javax.swing.JTextArea;

import genius.gui.panels.VflowPanelWithBorder;

/**
 * Shows the progress in a multiparty single session run
 */
@SuppressWarnings("serial")
public class SessionProgressUI extends VflowPanelWithBorder {

	private JSplitPane verticalsplit; // split on y axis
	private JSplitPane horizontalsplit; // split on x axis

	/**
	 * 
	 * @param model
	 *            an {@link OutcomesListModel} of this session.
	 * @param showChart
	 *            true iff any chart (normal or bi) has to be shown
	 * @param useBiChart
	 *            if true, a {@link SessionProgressUI} will be used that shows
	 *            utils of first two participants in util-util graph. Ignored if
	 *            showChart is false.
	 * @param textmodel
	 *            a {@link ActionDocumentModel}
	 */
	public SessionProgressUI(OutcomesListModel model, ActionDocumentModel textmodel, boolean showChart,
			boolean useBiChart, boolean showAllBids) {
		super("Session Progress");

		createSplitPanes();
		VflowPanelWithBorder textarea = new VflowPanelWithBorder("Negotiation log");
		textarea.add(new JScrollPane(new JTextArea(textmodel)));
		horizontalsplit.setLeftComponent(textarea);
		verticalsplit.setRightComponent(new JScrollPane(new JTable(new OutcomesModelToTableModelAdapter(model))));
		JPanel chart;
		if (showChart) {
			chart = useBiChart ? new ProgressChartBi(model, showAllBids) : new ProgressChart(model);
		} else {
			chart = new JPanel();
		}
		verticalsplit.setLeftComponent(chart);
		add(horizontalsplit);

	}

	/**
	 * Creates overall layout of this panel: a panel with a vertical split, with
	 * in the right side a panel with a horizontal split .
	 */
	private void createSplitPanes() {
		horizontalsplit = new JSplitPane();
		horizontalsplit.setOrientation(JSplitPane.HORIZONTAL_SPLIT);

		verticalsplit = new JSplitPane();
		verticalsplit.setOrientation(JSplitPane.VERTICAL_SPLIT);

		horizontalsplit.setRightComponent(verticalsplit);
	}

}
