package genius.core.representative;

import java.util.List;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

/** 
 * Utility of received and proposed bids is calculated as the weighted average of the utilities of the 
 * different preference profiles but according to a list of weights which show the inclination (belief) 
 * towards a particular preference profile 
 */

public class WeightedAverageFlatteningStrategy extends FlatteningStrategy{

	
	public WeightedAverageFlatteningStrategy(UncertainUtilitySpace uncertainpref){
		super(uncertainpref);
	}

	@Override 
	public double getUtility(Bid bid) {
		
		double sum=0;
		
		List<AbstractUtilitySpace> utilitySpaces = uncertainUtilitySpace.getUtilitySpaces();
		List<Double> weights = uncertainUtilitySpace.getWeights();


		for (int i = 0; i < utilitySpaces.size(); i++) {		
			try {
				// throws exception if bid incomplete or not in utility space
				} catch (Exception e) {
					e.printStackTrace();
					return 0;
				}
				sum += utilitySpaces.get(i).getUtility(bid) * weights.get(i);			
		}
		System.out.println("WeightedAverageUtil " +sum  +" ");

		return sum;				
	}

}
