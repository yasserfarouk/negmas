package agents.anac.y2012.MetaAgent.agents.WinnerAgent;

import java.util.ArrayList;
import java.util.Vector;

import genius.core.issue.ValueReal;

public class RealBinCreator extends BinCreator {
	double epsilon=0.01;
	@Override
	// creates the bins (ranges) for a continues issue - 
	// the minimum and maximum values are given 
	public ArrayList<DiscretisizedKey> createBins(double min, double max)
	{
		numOfBins = Math.max((int)(percentageOfRange*(max-min)), numConst);
		double binSize=(max-min)/numOfBins;
		ArrayList<DiscretisizedKey> bins=new ArrayList<DiscretisizedKey>();
		for (int i=0;i<numOfBins;i++)
		{
			DiscretisizedKey bin=new DiscretisizedKey(min, min+binSize);
			min=min+binSize;
			bins.add(bin);
		}
		epsilon=binSize/5; 
		return bins;
	}
	
	@Override
	// creates the vector of the values
	public Vector<ValueReal> createValuesVector(double min, double max)
	{
		// creating the bins
		ArrayList<DiscretisizedKey> bins = createBins(min,max);
		Vector<ValueReal> vectorOfVals = new Vector<ValueReal>();
		// for each bin add the values of around min and max and the average
		for(DiscretisizedKey bin : bins)
		{
			vectorOfVals.add(new ValueReal(bin.getMin()+epsilon));
			vectorOfVals.add(new ValueReal(bin.getMax()-epsilon));
			vectorOfVals.add(new ValueReal((bin.getMax()+bin.getMin())/2));
		}
		return vectorOfVals;
	}
}
