package genius.gui.progress.session;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataItem;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import genius.core.parties.PartyWithUtility;

/**
 * Shows progress as a plot where each party's utility is set vertically against
 * rounds horizontally. This assumes that only items are being added to the
 * OutcomesModel, one at a time.
 */
@SuppressWarnings("serial")
public class ProgressChart extends JPanel {

	final XYSeriesCollection dataset = new XYSeriesCollection();
	private OutcomesListModel model;
	private List<XYSeries> datasets = new ArrayList<XYSeries>();

	public ProgressChart(OutcomesListModel model) {
		this.model = model;
		setLayout(new BorderLayout());
		setMinimumSize(new Dimension(300, 300));

		for (PartyWithUtility party : model.getParties()) {
			XYSeries series = new XYSeries(party.getID().toString());
			datasets.add(series);
			dataset.addSeries(series);
		}

		JFreeChart xylineChart = ChartFactory.createXYLineChart("", "Round", "Utility", dataset,
				PlotOrientation.VERTICAL, true, true, false);

		ChartPanel chartPanel = new ChartPanel(xylineChart);
		final XYPlot plot = xylineChart.getXYPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		plot.setRenderer(renderer);

		// BAD: JFreeChart doesn't have a model so we must connect it directly
		model.addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {
			}

			@Override
			public void intervalAdded(ListDataEvent e) {
				addNewOutcome();
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
			}

		});

		add(chartPanel, BorderLayout.CENTER);
	}

	private void addNewOutcome() {
		SwingUtilities.invokeLater(new Runnable() {

			@Override
			public void run() {
				Outcome outcome = model.get(model.getSize() - 1);// last
				for (int n = 0; n < datasets.size(); n++) {
					// FIXME use datasets.get(n).getMaximumItemCount() or so
					datasets.get(n).add(new XYDataItem((double) outcome.getRound(),
							(double) outcome.getDiscountedUtilities().get(n)));
				}
			}
		});
	}

}
