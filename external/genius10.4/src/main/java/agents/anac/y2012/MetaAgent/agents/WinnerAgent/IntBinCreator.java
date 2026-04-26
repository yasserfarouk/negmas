package agents.anac.y2012.MetaAgent.agents.WinnerAgent;

import java.util.ArrayList;
import java.util.Vector;

import genius.core.issue.ValueInteger;

public class IntBinCreator extends BinCreator {
	@Override
	// creates the bins (ranges) for a continues issue - 
	// the minimum and maximum values are given 
	public ArrayList<DiscretisizedKey> createBins(double min, double max) {
		int nextInterval=1;
		numConst=Math.min(numConst, (int)(max-min));
		numOfBins = Math.max((int)(percentageOfRange*(max-min)), numConst);
		int binSize = (int)((max-min)/numOfBins);
		ArrayList<DiscretisizedKey> bins = new ArrayList<DiscretisizedKey>();
		if (binSize == 1)
			nextInterval = 0;
		for (int i=0;i<numOfBins;i++)
		{
			DiscretisizedKey bin=new DiscretisizedKey(min, Math.min(min+binSize,max));
			min=min+binSize+nextInterval;
			bins.add(bin);
		}
		return bins;
	}

	@Override
	// creates the vector of the values
	public Vector<ValueInteger> createValuesVector(double min, double max) {
		// creating the bins
		ArrayList<DiscretisizedKey> bins = createBins(min,max);
		Vector<ValueInteger> vectorOfVals = new Vector<ValueInteger>();
		boolean takeMin = true;
		// for each bin add the values of around min and max and the average
		for(DiscretisizedKey bin:bins)
		{
			if(takeMin) 
			{
				vectorOfVals.add(new ValueInteger((int)bin.getMin()));
			}
			else 
			{
				vectorOfVals.add(new ValueInteger((int)bin.getMax()));
			}
			vectorOfVals.add(new ValueInteger((int)(bin.getMax()+bin.getMin())/2));
			takeMin = !takeMin;
		}
		return vectorOfVals;
	}
}
