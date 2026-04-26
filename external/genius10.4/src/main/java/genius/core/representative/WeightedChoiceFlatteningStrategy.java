package genius.core.representative;

import java.util.List;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

/** 
 * One utility space is picked out of a distribution of utility spaces. 
 * This particular utility space is used throughout the negotiation.
 */

public class WeightedChoiceFlatteningStrategy extends RandomFlatteningStrategy {


	public final AbstractUtilitySpace selectedUtilitySpace;

	public WeightedChoiceFlatteningStrategy( UncertainUtilitySpace uncertainUtilitySpace) {
		super(uncertainUtilitySpace);
		this.selectedUtilitySpace = this.selectRandomUtilitySpace(uncertainUtilitySpace.getUtilitySpaces()
				, uncertainUtilitySpace.getWeights());
	}
	 
	
	@Override
	public double getUtility(Bid bid) {
	
		return this.selectedUtilitySpace.getUtility(bid);				
	}
	
}


