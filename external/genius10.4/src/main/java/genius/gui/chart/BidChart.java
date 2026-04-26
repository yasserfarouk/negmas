package genius.gui.chart;

import java.awt.Color;
import javax.swing.SwingUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultXYDataset;

public class BidChart {
	
	private double [][] possibleBids_;
	private double [][] pareto_;
	private double [][] bidSeriesA_;
	private double [][] bidSeriesB_;
	private double [][] lastBidA_;
	private double [][] lastBidB_;
	private double [][] nashPoint_;
	private double [][] kalaiPoint_;
	private double [][] rvA_;
	private double [][] rvB_;
	private double [][] agreement_;
	private String agentAName = "Agent A";
	private String agentBName = "Agent B";
	private JFreeChart chart;
	private XYPlot plot;
	private DefaultXYDataset possibleBidData = new DefaultXYDataset();
	private DefaultXYDataset paretoData = new DefaultXYDataset();
	private DefaultXYDataset bidderAData = new DefaultXYDataset();
	private DefaultXYDataset bidderBData = new DefaultXYDataset();
	private DefaultXYDataset bidderAReservationValueData = new DefaultXYDataset();
	private DefaultXYDataset bidderBReservationValueData = new DefaultXYDataset();
	private DefaultXYDataset nashData = new DefaultXYDataset();
	private DefaultXYDataset kalaiData = new DefaultXYDataset();
	private DefaultXYDataset agreementData = new DefaultXYDataset();
	private DefaultXYDataset lastBidAData = new DefaultXYDataset();
	private DefaultXYDataset lastBidBData = new DefaultXYDataset();
	final XYDotRenderer dotRenderer = new XYDotRenderer();
	final XYDotRenderer nashRenderer = new XYDotRenderer();
	final XYDotRenderer kalaiRenderer = new XYDotRenderer();
	final XYDotRenderer agreementRenderer = new XYDotRenderer();
	//final XYItemRenderer agreementRenderer = new XYLineAndShapeRenderer(false, true);
	final XYDotRenderer lastBidARenderer = new XYDotRenderer();
	final XYDotRenderer lastBidBRenderer = new XYDotRenderer();
	final XYItemRenderer paretoRenderer = new XYLineAndShapeRenderer(true,false);
	final XYItemRenderer reservationValueRenderer = new XYLineAndShapeRenderer(true,false);
	final XYItemRenderer lineARenderer = new XYLineAndShapeRenderer();
	final XYItemRenderer lineBRenderer = new XYLineAndShapeRenderer();
	private NumberAxis domainAxis;
    private ValueAxis rangeAxis;

	//empty constructor; but: don't you always know the possible bids and the pareto before the 1st bid? 
	public BidChart(){

		BidChart1();
		
	}
	public BidChart(String agentAname, String agentBname, double [][] possibleBids,double[][] pareto){		
		this.agentAName = agentAname;
		this.agentBName = agentBname;
		setPareto(pareto);
		setPossibleBids(possibleBids);
		BidChart1();
	}
	public void BidChart1(){
		chart = createOverlaidChart();  
		NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setRange(0,1.1);

		NumberAxis domainAxis = (NumberAxis)plot.getDomainAxis(); 
		domainAxis.setRange(0,1.1);
	}
	//returning the chart 
	public JFreeChart getChart(){
		return chart;
	}
	
	//set-Methods
	public void setPareto(double [][] pareto){
		this.pareto_ = pareto;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				paretoData.addSeries("Pareto efficient frontier",pareto_);
		    }
		});	
	}
	
	public void setPossibleBids(double [][] possibleBids){
		this.possibleBids_ = possibleBids;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
		    	possibleBidData.addSeries("all possible bids",possibleBids_);
		    }
		});	
	}
	
	public void setLastBidAData(double [][] lastBid)
	{
		this.lastBidA_ = lastBid;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
		    	lastBidAData.addSeries("Last bid by A", lastBidA_);
		    }
		});
	}
	
	public void setLastBidBData(double [][] lastBid)
	{
		this.lastBidB_ = lastBid;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
		    	lastBidAData.addSeries("Last bid by B", lastBidB_);
		    }
		});
	}
	
	public void setBidSeriesA(double [][] bidSeriesA){
		this.bidSeriesA_ = bidSeriesA;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				bidderAData.addSeries("Agent A's bids",bidSeriesA_);
		    }
		});		
	}
        
	public void setBidSeriesB(double [][] bidSeriesB) {
		this.bidSeriesB_ = bidSeriesB;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				bidderBData.addSeries("Agent B's bids",bidSeriesB_);
		    }
		});		
	}
	
	public void setBidderAReservationValue(double [][] bidderAReservationValue) {
		this.rvA_ = bidderAReservationValue;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				bidderAReservationValueData.addSeries("Agent A's reservation value", rvA_);  
		    }
		}); 
	}
        
	public void setBidderBReservationValue(double [][] bidderBReservationValue) {
		this.rvB_ = bidderBReservationValue;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				bidderBReservationValueData.addSeries("Agent B's reservation value", rvB_);  
		    }
		}); 
	}
	
	public void setNash(double[][] nash){
		this.nashPoint_ = nash;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				nashData.addSeries("Nash Point",nashPoint_);
		    }
		});	
	}
	public void setKalai(double[][] kalai){
		this.kalaiPoint_ = kalai;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
				nashData.addSeries("Kalai Point",kalaiPoint_);
		    }
		});	
	}
	public void setAgreementPoint(double[][]agreement){
		this.agreement_ = agreement;
		SwingUtilities.invokeLater(new Runnable() {
		    public void run() {
		    	agreementData.addSeries("Agreement",agreement_);
		    }
		});	
	}
			
	/**
     * Creates an overlaid chart.
     *
     * @return The chart.
     */
    private JFreeChart createOverlaidChart() {
    	domainAxis = new NumberAxis(agentAName);
        rangeAxis = new NumberAxis(agentBName);
        dotRenderer.setDotHeight(2);
        dotRenderer.setDotWidth(2);
		reservationValueRenderer.setSeriesPaint(0,Color.GRAY);
        nashRenderer.setDotHeight(5);
        nashRenderer.setDotWidth(5);
        nashRenderer.setSeriesPaint(0,Color.black);
        kalaiRenderer.setDotHeight(5);
        kalaiRenderer.setDotWidth(5);
        kalaiRenderer.setSeriesPaint(0,Color.pink);
        paretoRenderer.setSeriesPaint(0, Color.RED);
        lineARenderer.setSeriesPaint(0, Color.GREEN);
        lineBRenderer.setSeriesPaint(0, Color.BLUE);
        agreementRenderer.setDotHeight(10);
        agreementRenderer.setDotWidth(10);
        //agreementRenderer.setSeriesShape(0, new Ellipse2D.Float(10.0f, 10.0f, 10.0f, 10.0f));
        agreementRenderer.setSeriesPaint(0, Color.RED);
        lastBidARenderer.setSeriesPaint(0, Color.YELLOW);
        lastBidARenderer.setDotHeight(3);
        lastBidARenderer.setDotWidth(3);
        lastBidBRenderer.setSeriesPaint(0, Color.ORANGE);
        lastBidBRenderer.setDotHeight(3);
        lastBidBRenderer.setDotWidth(3);
       
        // createFrom plot ...
    	plot = new XYPlot(possibleBidData, domainAxis, rangeAxis, dotRenderer);
    	plot.setDataset(2, paretoData);
        plot.setRenderer(2, paretoRenderer);
        		
    	plot.setDataset(3, bidderAData);
	    plot.setRenderer(3, lineARenderer);
	    plot.setDataset(4, bidderBData);
	    plot.setRenderer(4, lineBRenderer);
	   
	    plot.setDataset(5, nashData);
	    plot.setRenderer(5, nashRenderer);
	    plot.setDataset(6, kalaiData);
	    plot.setRenderer(6, kalaiRenderer);
	    plot.setDataset(7, agreementData);
	    plot.setRenderer(7, agreementRenderer);
	    plot.setDataset(8, lastBidAData);
	    plot.setRenderer(8, lastBidARenderer);
	    plot.setDataset(9, lastBidBData);
	    plot.setRenderer(9, lastBidBRenderer);
		plot.setDataset(10, bidderAReservationValueData);
	    plot.setRenderer(10, reservationValueRenderer);
	    plot.setDataset(11, bidderBReservationValueData);
	    plot.setRenderer(11, reservationValueRenderer);
        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);
        // return a new chart containing the overlaid plot...
        JFreeChart chart = new JFreeChart("", JFreeChart.DEFAULT_TITLE_FONT, plot, true);
        chart.setBackgroundPaint(new Color(255,255,255));
        return chart;
    }
    public void setAgentAName (String value) {
    	agentAName = value;
    	domainAxis.setLabel(agentAName);
    }
    public void setAgentBName (String value) {
    	agentBName = value;
    	rangeAxis.setLabel(agentBName);
    }
}
