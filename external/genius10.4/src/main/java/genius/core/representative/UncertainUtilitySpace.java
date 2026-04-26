package genius.core.representative;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.OutcomeSpace;
import genius.core.misc.Range;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UtilitySpace;
import genius.core.xml.SimpleElement;
	
	/**
	 * Μajor Class incorporating multiple preference profiles introducing uncertainty about user preferences
	 * created by D.Tsimpoukis, Jan 2018
	 */

public class UncertainUtilitySpace extends AbstractUtilitySpace{
	
	private static final long serialVersionUID = 2981354315582885764L;
	
	/** All possible utility spaces considered */
	protected List<AbstractUtilitySpace> utilitySpaces;	
	/** The corresponding normalized weights of each utility space */
	protected List<Double> weights;
	
	protected FlatteningStrategy flatteningStrategy;	
	
	
	/**
	 * Automatically normalizes the weights
	 * @param uspaces
	 * @param dom
	 * @param weights
	 */
	public UncertainUtilitySpace(List<AbstractUtilitySpace> uspaces , Domain dom, double [] weights)
	{
		super(dom);
		this.utilitySpaces = uspaces;
		
		double [] normalizedWeights = new double [weights.length];
		normalizedWeights = normalizeWeights(weights);
		this.weights = new ArrayList<Double>();
		
		for (int i = 0; i < uspaces.size(); i++){
			this.weights.add(normalizedWeights[i]);
		}	
	}
	
	public UncertainUtilitySpace (List<AbstractUtilitySpace> uspaces, Domain dom) {
		
		this(uspaces, dom, createUniformWeights(uspaces.size()));
	}

		
	public UncertainUtilitySpace(List<AbstractUtilitySpace> uspaces, Domain dom, String flatteningStrategy){
		this(uspaces , dom);
		setFlatteningStrategyByName(flatteningStrategy);		
	}
	
	public UncertainUtilitySpace(List<AbstractUtilitySpace> uspaces , Domain dom, double [] weights , FlatteningStrategy flatteningStrategy){
		this(uspaces , dom , weights);
		setFlatteningStrategy(flatteningStrategy);
	}
	
	public UncertainUtilitySpace(List<AbstractUtilitySpace> uspaces , Domain dom, double [] weights , String flatteningStrategy){
		this(uspaces , dom , weights);
		setFlatteningStrategyByName(flatteningStrategy);
	}
	
	public UncertainUtilitySpace (UncertainUtilitySpace multiPrefUS){
		super(multiPrefUS.getDomain());
		this.utilitySpaces = multiPrefUS.getUtilitySpaces();
		this.weights = multiPrefUS.getWeights();
	}
	
	/**
	 * Method used to normalize the array of imported weights and scale them between [0,1]	
	 */	
	private static double [] normalizeWeights (double [] ws) {
		double [] normalizedWeights = new double[ws.length];
		double sum = 0;
		
		for (double x : ws) {
			sum += x;
		}
		
		for (int i = 0; i < ws.length; i++) {
			normalizedWeights[i] = ws[i]/sum;
		}
		return normalizedWeights;
	}
	
	private static double [] createUniformWeights(int size) 
	{
		double [] weights = new double [size];	
		for ( int i = 0; i < size; i++) {
			weights[i] = 1.0 / size;
		}
		return weights;
	}
	
	public static List<Double> createUniformWeightsList(int size)
	{
		ArrayList<Double> weights = new ArrayList<Double>();
		for (int i = 0; i < size; i++) {
			weights.add(1.0 / size);			
		}
		return weights;
	}

	/** Method used to flatten the list of utility spaces. It is used to simulate the creation of a new utility space
	 * according to a certain mapping of outcomes to utilities. An example is the Average strategy in which case the 
	 * utility of a bid is calculated as the average of the bid utility of each one of the utility spaces on the list.
	*/
	
	public final void setFlatteningStrategyByName (String flatteningStrategy) {
		
		switch (flatteningStrategy) {
		
			case "RandomFlattening" : this.flatteningStrategy = new RandomFlatteningStrategy(this);	
				break;
			case "AverageFlattening" : this.flatteningStrategy = new AverageFlatteningStrategy(this);
				break;
			case "WeightedAverageFlattening" : this.flatteningStrategy = new WeightedAverageFlatteningStrategy(this);
				break;
			case "WeightedChoiceFlattening" : this.flatteningStrategy = new WeightedChoiceFlatteningStrategy(this);	
				break;
			default : System.out.println ("No such flattening strategy exists. Set a correct flattening strategy");
				break;
		}	
	}

	
	@Override
	public double getUtility(Bid bid) {

			return flatteningStrategy.getUtility(bid);	
	}

	/** Returns a list of the size of the number of utility spaces each element of which is a list
	 * of the bids in the range we want
	 */
	
	public List<List<BidDetails>> getBidsInRangeByUtilitySpace(Range range)
	{
		List<OutcomeSpace> outcomeSpaces = new ArrayList<OutcomeSpace>();
		for (AbstractUtilitySpace us : utilitySpaces)
			outcomeSpaces.add(new OutcomeSpace(us));
		
		List<List<BidDetails>> multiPrefBidsInRange = new ArrayList<List<BidDetails>> ();
		
		for (OutcomeSpace os : outcomeSpaces)	
			multiPrefBidsInRange.add(os.getBidsinRange(range));			
		
		return multiPrefBidsInRange;		
	}
	

	@Override
	public UtilitySpace copy() {
		return new UncertainUtilitySpace(this);
	}
	

	@Override
	public String isComplete() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SimpleElement toXML() throws IOException {
		// TODO Auto-generated method stub
		return null;
	}
	
	public List<AbstractUtilitySpace> getUtilitySpaces() {
		return utilitySpaces;
	}

	public void setUtilitySpaces(List<AbstractUtilitySpace> uspaces) {
		this.utilitySpaces = uspaces;
	}

	public List<Double> getWeights() {
		return weights;
	}

	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}
	
	public FlatteningStrategy getFlatteningStrategy() {
		return flatteningStrategy;
	}
	
	public void setFlatteningStrategy(FlatteningStrategy flatteningStrategy) {
		this.flatteningStrategy = flatteningStrategy;
	}
	
}
	



	


