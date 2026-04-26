package genius.core.representative;

import genius.core.Bid;

/**
 *  Abstract class used to describe the flattening strategy of agents that incorporate multiple preference profiles.
 *  Flattening is a strategy to simplify the uncertain utility space.
 */

public abstract class FlatteningStrategy {
	
	protected UncertainUtilitySpace uncertainUtilitySpace;
	
	FlatteningStrategy (UncertainUtilitySpace uncertainUtilitySpace) {
		this.uncertainUtilitySpace = uncertainUtilitySpace;		
	}
	
	public abstract double getUtility(Bid bid);
	
	public String getName() {
		return this.getClass().getSimpleName();
	}
}
