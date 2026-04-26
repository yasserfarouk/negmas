package genius.core.representative;

import java.util.ArrayList;
import java.util.List;

import genius.core.utility.AbstractUtilitySpace;

public class AverageFlatteningStrategy extends WeightedAverageFlatteningStrategy{
	

	public AverageFlatteningStrategy(UncertainUtilitySpace uncertainUtilitySpace){
		super(uncertainUtilitySpace);
		uniformWeights();
	}
	
	private void uniformWeights() 
	{		
		uncertainUtilitySpace.setWeights(UncertainUtilitySpace.createUniformWeightsList(
				uncertainUtilitySpace.getUtilitySpaces().size()));
	}
	
}
