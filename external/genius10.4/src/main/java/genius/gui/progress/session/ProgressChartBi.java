package genius.gui.progress.session;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataItem;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.analysis.BidPoint;
import genius.core.analysis.MultilateralAnalysis;
import genius.core.utility.UtilitySpace;

/**
 * * Bilateral version of ProgressChart. Shows utility of party 1 on the X axis,
 * and of party 2 on the Y (vertical) axis.
 */
@SuppressWarnings("serial")
public class ProgressChartBi extends JPanel {

	private static final String UTILITY = " utility";
	final XYSeriesCollection dataset = new XYSeriesCollection();
	private OutcomesListModel model;
	private XYSeries bidsA, bidsB, agreements;
	private AgentID party1;
	private AgentID party2;

	/**
	 * @param model
	 *            all outcomes so far. Assumes that only items are being added
	 *            to the OutcomesModel, one at a time. Assumes that model has 2
	 *            parties.
	 * @param showAllBids
	 *            all bids in the bidspace are shown iff this is true.
	 */
	public ProgressChartBi(OutcomesListModel model, boolean showAllBids) {
		this.model = model;
		setLayout(new BorderLayout());
		setMinimumSize(new Dimension(300, 300));
		party1 = model.getParties().get(0).getID();
		party2 = model.getParties().get(1).getID();
		NumberAxis domainAxis = new NumberAxis(party1.toString() + UTILITY);
		NumberAxis rangeAxis = new NumberAxis(party2.toString() + UTILITY);

		bidsA = new XYSeries(party1.toString(), false);
		bidsB = new XYSeries(party2.toString(), false);
		agreements = new XYSeries("agreements");
		XYPlot plot = new XYPlot(dataset, domainAxis, rangeAxis, new XYLineAndShapeRenderer());

		dataset.addSeries(agreements); // first one has priority on plot.
		dataset.addSeries(bidsA);
		dataset.addSeries(bidsB);

		// add some background info
		MultilateralAnalysis analysis = new MultilateralAnalysis(model.getParties(), null, null);
		dataset.addSeries(getNash(analysis));
		dataset.addSeries(getKalai(analysis));
		dataset.addSeries(getPareto(analysis));

		if (showAllBids) {
			XYDotRenderer dotRenderer = new XYDotRenderer();
			dotRenderer.setDotHeight(2);
			dotRenderer.setDotWidth(2);
			dotRenderer.setSeriesPaint(0, Color.PINK);
			plot.setDataset(2, getAllBids());
			plot.setRenderer(2, dotRenderer);
		}

		// fix the color of renderer 3: its color is invisible yellow ...
		plot.getRenderer().setSeriesPaint(3, Color.BLACK);

		JFreeChart freechart = new JFreeChart("", JFreeChart.DEFAULT_TITLE_FONT, plot, true);
		ChartPanel chartPanel = new ChartPanel(freechart);

		/**
		 * BAD: JFreeChart doesn't have a model so we must connect it directly.
		 * actually, there IS a model: {@link XYDataset}. Maybe we can fix this.
		 */
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

	private XYSeries getKalai(MultilateralAnalysis analysis) {
		XYSeries kalaiSeries = new XYSeries("Kalai");
		BidPoint point = analysis.getKalaiPoint();
		kalaiSeries.add(point.getUtilityA(), point.getUtilityB());
		return kalaiSeries;
	}

	private XYSeries getNash(MultilateralAnalysis analysis) {
		XYSeries nashSeries = new XYSeries("Nash");
		BidPoint point = analysis.getNashPoint();
		nashSeries.add(point.getUtilityA(), point.getUtilityB());
		return nashSeries;
	}

	/**
	 * Get all bids in a dataset. Just iterates over domain of first party
	 * 
	 * @return {@link XYDataSet} containing all bids.
	 */
	private XYDataset getAllBids() {

		UtilitySpace utils1 = model.getParties().get(0).getUtilitySpace();
		UtilitySpace utils2 = model.getParties().get(1).getUtilitySpace();

		BidIterator bids = new BidIterator(utils1.getDomain());
		XYSeries series = new XYSeries("All bids");
		while (bids.hasNext()) {
			Bid bid = bids.next();
			series.add(utils1.getUtility(bid), utils2.getUtility(bid));
		}
		return new XYSeriesCollection(series);
	}

	private XYSeries getPareto(MultilateralAnalysis analysis) {
		XYSeries series = new XYSeries("pareto");
		for (BidPoint point : analysis.getParetoFrontier()) {
			series.add(point.getUtilityA(), point.getUtilityB());
		}

		return series;
	}

	private void addNewOutcome() {
		SwingUtilities.invokeLater(new Runnable() {

			@Override
			public void run() {
				// invariant n is current number of data points in graph.
				int n = bidsA.getItemCount() + bidsB.getItemCount();

				// continue till all model points are in graph
				while (n < model.size()) {
					Outcome outcome = model.get(n);
					XYDataItem item = new XYDataItem((double) outcome.getDiscountedUtilities().get(0),
							(double) outcome.getDiscountedUtilities().get(1));
					if (party1.equals(outcome.getAgentID())) {
						bidsA.add(item);
					} else {
						bidsB.add(item);
					}
					if (outcome.isAgreement()) {
						agreements.add(item);
					}
					n++;
				}

			}
		});
	}

}
