package agents.anac.y2015.JonnyBlack;

import java.util.Collections;
import java.util.Vector;

import genius.core.Bid;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

import java.util.HashMap;

public class Party {
	String ID;
	int counts[][];
	double weights[];
	public Vector<BidHolder> orderedBids;
	Party(String id,int[][] issueVals)
	{
		this.ID =id;
		counts = new int[issueVals.length][];
		for(int i=0;i<issueVals.length;i++)
		{
			counts[i] = new int[issueVals[i].length];
		}
		weights = new double[issueVals.length];
		this.orderedBids=new Vector<BidHolder>();
	}
	
	public double getPredictedUtility(Bid b,AdditiveUtilitySpace us)
	{
		double total = 0;
		for(int i=0;i<this.counts.length;i++)
		{
			total += getValueForIssue(b, us, i);
		}
		return total;
	}
	
	public double getValueForIssue(Bid b, AdditiveUtilitySpace us,int i)
	{
		double eval =0;
		try {
		IssueDiscrete id = (IssueDiscrete)us.getIssue(i);
		int choice = id.getValueIndex((ValueDiscrete)b.getValue(i+1));
		eval = weights[i]*getVal(i, choice);
				
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return eval;
	}
	
	public double getVal(int i,int j)
	{
		int ord=0;
		for(int k =0;k<counts[i].length;k++)
		{
			if(counts[i][k]<=counts[i][j])
				ord++;
		}
		double part = 1.0/counts[i].length;
		return ord*part;
	}
	
	public void calcWeights()
	{
		calcWeightsWithGini();
		//calcWeightsWithEntropy();
	}
	
	public void calcWeightsWithGini()
	{
		double sumweight =0;
		for(int i =0;i<weights.length;i++)
		{
			this.weights[i] = weigthWithGini(i);
			sumweight+=this.weights[i];
		}
		for(int i =0;i<weights.length;i++)
		{
			this.weights[i] /= sumweight;
		
		}
	}
	
	
	public void calcWeightsWithEntropy()
	{
		double sumweight =0;
		double maxent = 0;
		for(int i =0;i<weights.length;i++)
		{
			this.weights[i] = weigthWithEntropy(i);
			if(this.weights[i]>maxent)
				maxent=this.weights[i];
		}
		for(int i =0;i<weights.length;i++)
		{
			if(maxent==0)
				this.weights[i]=1;
			else
				this.weights[i]=maxent/this.weights[i];
			sumweight+=this.weights[i];
		}
		for(int i =0;i<weights.length;i++)
		{
			this.weights[i] /= sumweight;
		
		}
	}
	
	public double getWeight(int ind)
	{
		return this.weigthWithGini(ind);
	}
	
	private double weigthWithGini(int ind)
	{
		double w = 0;
		int sum =0;
		for(int i =0;i<this.counts[ind].length;i++)
		{
			sum+=this.counts[ind][i];
			double r =  1.0*this.counts[ind][i]*this.counts[ind][i];
			w+=r;
		}
		return w/(sum*sum);
	}
	
	private double weigthWithEntropy(int ind)
	{
		double w = 0;
		int sum =0;
		for(int i =0;i<this.counts[ind].length;i++)
		{
			sum+=this.counts[ind][i];
		}
		for(int i =0;i<this.counts[ind].length;i++)
		{
			if(this.counts[ind][i]==0)
				continue;
			double p = 1.0*this.counts[ind][i]/sum;
			w+= - p*Math.log(p);
		}
		return w/(sum*sum);
	}
	@Override
	public boolean equals(Object obj) {
		return this.ID.equals(((Party) obj).ID);
	}
	
	public void show()
	{	
		for(int i =0;i<this.counts.length;i++)
		{
			System.out.print(this.weights[i]+" ");
			for(int j =0;j<this.counts[i].length;j++)
			{
				System.out.print(this.counts[i][j]+" ");
			}	
			System.out.println();
		}
		
	}	
	public void orderBids(AdditiveUtilitySpace us)
	{
		for(BidHolder b : orderedBids)
		{
			b.v = this.getPredictedUtility(b.b, us);
		}
		Collections.sort(orderedBids);
	}

	public void setOrderedBids(Vector<BidHolder> acceptableBids,AdditiveUtilitySpace us) {
		for(BidHolder bh :acceptableBids)
		{
			BidHolder bh1 = new BidHolder();
			bh1.b=bh.b;
			bh1.v = this.getPredictedUtility(bh.b, us);
			this.orderedBids.add(bh1);
		}
	}
}
