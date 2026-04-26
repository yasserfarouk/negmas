package genius.core.representative;

import java.util.List;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class RandomFlatteningStrategy extends FlatteningStrategy{
	
		public RandomFlatteningStrategy( UncertainUtilitySpace uncertainUtilitySpace) {
			super(uncertainUtilitySpace);
		}
		 
		
		@Override
		public double getUtility(Bid bid) {
		
			AbstractUtilitySpace randomlySelectedUtilitySpaceForBidding;
			randomlySelectedUtilitySpaceForBidding = selectRandomUtilitySpace(
					this.uncertainUtilitySpace.getUtilitySpaces() , this.uncertainUtilitySpace.getWeights());
			return randomlySelectedUtilitySpaceForBidding.getUtility(bid);				
		}
		
		public AbstractUtilitySpace selectRandomUtilitySpace(List<AbstractUtilitySpace> uspaces , List<Double> weights) {
			RandomCollection <AbstractUtilitySpace> items = new RandomCollection<AbstractUtilitySpace>();
			
			for (int i=0; i < uspaces.size(); i++) {
				items.add(weights.get(i), uspaces.get(i));						
			}
			
			return items.next();	
		}

	}


