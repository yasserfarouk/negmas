package agents.anac.y2019.fsega2019.fsegaoppmodel;


import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.UtilitySpace;



public class MyBayesianOpponentModel extends OpponentModel
{
	private AbstractUtilitySpace uUS;
	private ArrayList<UtilitySpaceHypothesis> uUSHypothesis;
	private double previousBidUtility;
	private double SIGMA = 0.25;
	private double CONCESSION_STRATEGY = 0.035; //estimated opponent concession strategy
	private int initialNumberHyps = 0;

	//scalable
	private boolean	bUseMostProb = true;
	private ArrayList<UtilitySpaceHypothesis> mostProbHyps;
	
	public MyBayesianOpponentModel(AbstractUtilitySpace pUS)
	{
		//TODO: for test
		//System.out.println("Model creation");
		
		if(pUS == null)
			throw new NullPointerException("MyBayesianOpponentModel: utility space = null");
		uUS = pUS;
		
		previousBidUtility = 1;
		dDomain = pUS.getDomain();
		//aBiddingHistory = new ArrayList<Bid>();

		List<Issue> issues = dDomain.getIssues();
		ArrayList<ArrayList<EvaluatorHypothesis>> aaEvaluatorHypothesis = new ArrayList<ArrayList<EvaluatorHypothesis>> ();
		
		int numberOfIssues = issues.size();
		
		//generate weight hypothesis ==> <count of issues>! hypothesis
		WeightHypothesis[] weightHypothesis = new WeightHypothesis[factorial(numberOfIssues)];
		
		//create all permutations
		double[] P = new double[numberOfIssues];
		
		//normalize weights
		for(int i = 0; i < numberOfIssues; i++)
			P[i] = 2.0 * (i + 1) / (double)(numberOfIssues * (numberOfIssues + 1));
		weightPermutations(0, weightHypothesis, P, numberOfIssues - 1);
		
		//add initial probabilities
		for(int i = 0; i < weightHypothesis.length; i++)
			weightHypothesis[i].setProbability(1.0/weightHypothesis.length);

		//generate evaluator hypotheses
		for(int i = 0; i < numberOfIssues; i++)
		{
			ArrayList<EvaluatorHypothesis> lEvalHyps;
//			switch(uUS.getEvaluator(issues.get(i).getNumber()).getType())
//			{
//				case DISCRETE:
					lEvalHyps = new ArrayList<EvaluatorHypothesis>();
					aaEvaluatorHypothesis.add(lEvalHyps);
					IssueDiscrete lDiscIssue = (IssueDiscrete)(dDomain.getIssues().get(i)); 
					
					//uphill
					EvaluatorDiscrete lDiscreteEvaluator = new EvaluatorDiscrete();
					for(int j = 0; j < lDiscIssue.getNumberOfValues(); j++) 
						lDiscreteEvaluator.addEvaluation(lDiscIssue.getValue(j), 1000 * j + 1);
					EvaluatorHypothesis lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEvaluator, "uphill");
					
					lEvalHyps.add(lEvaluatorHypothesis);
					
					//downhill
					lDiscreteEvaluator = new EvaluatorDiscrete();
					for(int j=0; j < lDiscIssue.getNumberOfValues(); j++) 
						lDiscreteEvaluator.addEvaluation(lDiscIssue.getValue(j), 1000 * (lDiscIssue.getNumberOfValues() - j - 1) + 1);
					lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEvaluator, "downhill");
					
					lEvalHyps.add(lEvaluatorHypothesis);
					
					//triangular
					lDiscreteEvaluator = new EvaluatorDiscrete();
					int halfway=lDiscIssue.getNumberOfValues()/2;
					for(int j=0;j<lDiscIssue.getNumberOfValues();j++)
						if(j<halfway)
							lDiscreteEvaluator.addEvaluation(lDiscIssue.getValue(j),1000*j+1);
						else
							lDiscreteEvaluator.addEvaluation(lDiscIssue.getValue(j), 
									1000*(lDiscIssue.getNumberOfValues()-j-1)+1);
					lEvaluatorHypothesis = new EvaluatorHypothesis (lDiscreteEvaluator, "triangular");
					
					lEvalHyps.add(lEvaluatorHypothesis);
//					break;
//				
//				//Eval hypothesis for real / price attributes
//				case PRICE:
//				case REAL:
//					lEvalHyps = new ArrayList<EvaluatorHypothesis>();
//					aaEvaluatorHypothesis.add(lEvalHyps);
//					IssueReal lRealIssue = (IssueReal)(dDomain.getIssue(i));
//				
//					//uphill
//					EvaluatorReal lRealEvaluator = new EvaluatorReal();
//					lRealEvaluator.setLowerBound(lRealIssue.getLowerBound());
//					lRealEvaluator.setUpperBound(lRealIssue.getUpperBound());
//					lRealEvaluator.setType(EVALFUNCTYPE.LINEAR);
//					lRealEvaluator.addParam(1, 1.0/(lRealEvaluator.getUpperBound()-lRealEvaluator.getLowerBound()));
//					lRealEvaluator.addParam(0, -lRealEvaluator.getLowerBound()/(lRealEvaluator.getUpperBound()-lRealEvaluator.getLowerBound()));
//					lEvaluatorHypothesis = new EvaluatorHypothesis(lRealEvaluator, "uphill");
//					lEvalHyps.add(lEvaluatorHypothesis);
//					
//					//downhill
//					lRealEvaluator = new EvaluatorReal();
//					lRealEvaluator.setLowerBound(lRealIssue.getLowerBound());
//					lRealEvaluator.setUpperBound(lRealIssue.getUpperBound());
//					lRealEvaluator.setType(EVALFUNCTYPE.LINEAR);
//					lRealEvaluator.addParam(1, -1.0/(lRealEvaluator.getUpperBound()-lRealEvaluator.getLowerBound()));
//					lRealEvaluator.addParam(0, 1.0+lRealEvaluator.getLowerBound()/(lRealEvaluator.getUpperBound()-lRealEvaluator.getLowerBound()));
//					lEvaluatorHypothesis = new EvaluatorHypothesis(lRealEvaluator, "downhill");
//					lEvalHyps.add(lEvaluatorHypothesis);
//					
//					//triangular
//					int lTotalTriangularFns = 1;
//					for(int k=1;k<=lTotalTriangularFns;k++) 
//					{
//						lRealEvaluator = new EvaluatorReal();
//						lRealEvaluator.setLowerBound(lRealIssue.getLowerBound());
//						lRealEvaluator.setUpperBound(lRealIssue.getUpperBound());
//						lRealEvaluator.setType(EVALFUNCTYPE.TRIANGULAR);
//						lRealEvaluator.addParam(0, lRealEvaluator.getLowerBound());
//						lRealEvaluator.addParam(1, lRealEvaluator.getUpperBound());
//						lRealEvaluator.addParam(2, lRealEvaluator.getLowerBound()+(double)k*(lRealEvaluator.getUpperBound()-lRealEvaluator.getLowerBound())/(lTotalTriangularFns+1));
//						lEvaluatorHypothesis = new EvaluatorHypothesis(lRealEvaluator, "triangular");
//						lEvaluatorHypothesis.setProbability((double)1/3);
//						lEvalHyps.add(lEvaluatorHypothesis);
//					}
//					for(int k=0;k<lEvalHyps.size();k++) {
//						lEvalHyps.get(k).setProbability((double)1/lEvalHyps.size());
//					}
//					
//					break;
//					
//				default:
//					throw new NullPointerException("Evaluator type not implemented: eval type - " + uUS.getEvaluator(issues.get(i).getNumber()).getType());
			}
		//}
		
		//build evaluation hypothesis
		ArrayList<EvaluatorHypothesis[]> evalHypothesis = new ArrayList<EvaluatorHypothesis[]>();
//	uUS.getDomain().getIssues().size()
		
		EvaluatorHypothesis[] ehTmp = new EvaluatorHypothesis[uUS.getDomain().getIssues().size()];
	

		buildEvaluationHypothesis(evalHypothesis, ehTmp, uUS.getDomain().getIssues().size() - 1, aaEvaluatorHypothesis);
		
		//build user space hypothesis
		buildUtilitySpaceHypothesis(weightHypothesis, evalHypothesis);

		// HOSOKAWA: removed this part
//        try
//        {
//            FileWriter fw = new FileWriter("E:\\Hyopothesis.txt");
//            fw.write(uUSHypothesis.toString());
//            fw.flush();
//            fw.close();
//        }
//        catch(Exception e)
//        {  }
        initialNumberHyps = uUSHypothesis.size();
	}
	
	private void buildEvaluationHypothesis(ArrayList<EvaluatorHypothesis[]> pHyps, EvaluatorHypothesis[] pEval, int m, ArrayList<ArrayList<EvaluatorHypothesis>> paaEval)
	{
		if(m==0)
		{
			ArrayList<EvaluatorHypothesis> lEvalHyps = paaEval.get(uUS.getDomain().getIssues().size()-1);
			for (int i=0;i<lEvalHyps.size();i++)
			{
				pEval[uUS.getDomain().getIssues().size()-1] = lEvalHyps.get(i);
				EvaluatorHypothesis[] lTmp = new EvaluatorHypothesis[uUS.getDomain().getIssues().size()];
				//copy to temporary array
				for(int j =0;j<lTmp.length;j++)
					lTmp[j] = pEval[j]; 
				pHyps.add(lTmp);
			}
		}
		else
		{
			ArrayList<EvaluatorHypothesis> lEvalHyps = paaEval.get(uUS.getDomain().getIssues().size()-m-1);
			for (int i=0;i<lEvalHyps.size();i++)
			{
				pEval[uUS.getDomain().getIssues().size()-m-1] = lEvalHyps.get(i);
				buildEvaluationHypothesis(pHyps, pEval, m-1, paaEval);
			}			
		}
}
	
	private void buildUtilitySpaceHypothesis(WeightHypothesis[] pWeightHypothesis, ArrayList<EvaluatorHypothesis[]> pEvalHypothesis)
	{
		uUSHypothesis = new ArrayList<UtilitySpaceHypothesis>();
		for(int i = 0; i < pWeightHypothesis.length; i++)
		{
			for(int j=0; j < pEvalHypothesis.size(); j++) {
				UtilitySpaceHypothesis lUSHyp = new UtilitySpaceHypothesis(dDomain, uUS, pWeightHypothesis[i], pEvalHypothesis.get(j));
				uUSHypothesis.add(lUSHyp);
			}
		}
		
		//set initial probability for all hyps
		for(int i = 0; i < uUSHypothesis.size(); i++)
		{
			uUSHypothesis.get(i).setProbability(1.0/(double)(uUSHypothesis.size()));
		}		
	}
	
	private Integer weightPermutations(Integer index, WeightHypothesis[] hyps, double[] P, int m) {
		if(m == 0) {
			WeightHypothesis lWH = new WeightHypothesis(dDomain);
			for(int i = 0; i < P.length; i++) lWH.setWeight(i, P[i]);
			hyps[index] = lWH;
			index++;
		} else {
			for(int i = 0; i <= m; i++) {
				index = weightPermutations(index, hyps, P, m-1);
				if(i<m) {
					//swap elements i and m
					double tmp = P[i];
					P[i]=P[m];
					P[m] = tmp;
					reverse(P, m - 1);
				} //if
			}
		}
		return index;		
	}
	
	private void reverse(double[] array, int size)
	{
		int i=0, j=size;
		while(i < j)
		{
			//swap i <-> j
			double tmp = array[i];
			array[i] = array[j];
			array[j] = tmp;
			i++;
			j--;
		}
	}
	
	private int factorial(int n)
	{
		int result = 1;
		for(; n > 1; n--)
		{
			result *= n;
		}
		return result;
	}

	public void updateBeliefs(Bid pBid) throws Exception
	{
		//TODO: this variable counts the number of excludes
		int iExcludeCnt = 0;
		
		//calculate probability for the given bid
		double lProbSum = 0;
		double lMaxProb = 0;
		for(int i = 0; i < uUSHypothesis.size(); i++)
		{

			UtilitySpaceHypothesis hyp = uUSHypothesis.get(i);
			double condDistrib = hyp.getProbability() * conditionalDistribution(uUSHypothesis.get(i).getUtility(pBid), previousBidUtility);
			lProbSum += condDistrib;
			if(condDistrib > lMaxProb)
				lMaxProb = condDistrib;
			hyp.setProbability(condDistrib);
			

		}
	
		if(bUseMostProb)
			mostProbHyps = new ArrayList<UtilitySpaceHypothesis>();
		
		double mostProbHypSum = 0;
	

		//update the weights hyps and evaluators hyps
		for(int i = 0; i < uUSHypothesis.size(); i++)
		{

			UtilitySpaceHypothesis hyp = uUSHypothesis.get(i);
			double normalizedProbability = hyp.getProbability() / lProbSum; 

			if(bUseMostProb)
				if(normalizedProbability > lMaxProb * 0.95 / lProbSum) {
					mostProbHyps.add(hyp);
					mostProbHypSum += normalizedProbability;
				}

			//exclude if probability is 0
			if(normalizedProbability > 0)
				hyp.setProbability(normalizedProbability);
			else
			{
				iExcludeCnt++;
				uUSHypothesis.remove(i);
			}
			// --- end exclude hyps with prob. around 0
		}
		
		//normalize most probable hypothesis
		if(bUseMostProb) { 

			for(int i=0;i<mostProbHyps.size();i++) {			
				UtilitySpaceHypothesis tmpHyp = mostProbHyps.get(i);
				double normalizedProbability = tmpHyp.getProbability() / mostProbHypSum; 
				tmpHyp.setProbability(normalizedProbability);
			}
		}

		//TODO: print excluded hypothesis count - with 0
		//System.out.println("Excluded hypothesis with 0: " + iExcludeCnt);
		
		if(uUSHypothesis.size() / initialNumberHyps >= 0.15)
        {
		
		//calculate utility of the next partner's bid according to the concession functions
		previousBidUtility = previousBidUtility - CONCESSION_STRATEGY;
		
		//sort hypotesis by probability

		Collections.sort(uUSHypothesis);


		//exclude bids with sum under 0.95
		int cutPoint = Integer.MAX_VALUE;
		
		double cummulativeSum = 0;
		
		//get cutPoint
		//and cumulative sum for normalization
		for(int i = 0; i < uUSHypothesis.size(); i++)
		{
			cummulativeSum += uUSHypothesis.get(i).getProbability();
			if(cummulativeSum > 0.95)
			{
				cutPoint = i;
				break;
			}
		}
		//System.out.println("Cumulative sum: " + cummulativeSum);
		
		//eliminate from cutPoint to last item
		if(cutPoint != Integer.MAX_VALUE)
		{
			for(int i = uUSHypothesis.size() - 1; i >= cutPoint; i--)
			{
				iExcludeCnt++;
				uUSHypothesis.remove(i);
			}
		}
		
		double normalizedUtilSum = 0;
		
		//normalize remained hypothesis probability
		for(int i = 0; i < uUSHypothesis.size(); i++)
		{
			UtilitySpaceHypothesis currentHyp = uUSHypothesis.get(i);
			double newProbability = currentHyp.getProbability() / cummulativeSum;
			currentHyp.setProbability(newProbability);
			normalizedUtilSum += newProbability;
		}
		//TODO: print info about excludes
		System.out.println("Excluded hypothesis at this step: " + iExcludeCnt);
		System.out.println("Count of remained Hyps: " + uUSHypothesis.size());
		System.out.println("NormSum: " + normalizedUtilSum);
        }
	}
	
	private double conditionalDistribution(double pUtility, double pPreviousBidUtility)
	{
		if(pPreviousBidUtility < pUtility)
			return 0;
		else
		{
			double x = (pPreviousBidUtility - pUtility) / pPreviousBidUtility;
			double distribution = (1 / (SIGMA * Math.sqrt(2 * Math.PI)) * Math.exp(-(x * x) / (2 * SIGMA * SIGMA)));
			return distribution;
		}
	}
	
	public double getExpectedUtility(Bid pBid) throws Exception
	{
		double lExpectedUtility = 0;
		
		if(bUseMostProb &&(mostProbHyps != null)) {
			for(int i=0; i < mostProbHyps.size(); i++) {
				UtilitySpaceHypothesis tmpUSHypothesis = mostProbHyps.get(i);
				double p = tmpUSHypothesis.getProbability();
				double u = tmpUSHypothesis.getUtility(pBid);
				lExpectedUtility += p * u;
			}
		}
		else
		{		
			for(int i = 0; i < uUSHypothesis.size(); i++)
			{
				UtilitySpaceHypothesis tmpUSHypothesis = uUSHypothesis.get(i);
				double p = tmpUSHypothesis.getProbability();
				double u = tmpUSHypothesis.getUtility(pBid);
				lExpectedUtility += p * u;
			}
		}
		
		//TODO: for test
		//System.out.println("utility: " + lExpectedUtility + "for bid: " + pBid);
		
		return lExpectedUtility;	
	}
	
	public double getExpectedWeight(int pIssueNumber)
	{
		double lExpectedWeight = 0;
		for(int i=0; i < uUSHypothesis.size(); i++) {
			UtilitySpaceHypothesis lUSHyp = uUSHypothesis.get(i);
			double p = lUSHyp.getProbability();
			double u = lUSHyp.getHeightHyp().getWeight(pIssueNumber);
			lExpectedWeight += p * u;
		}
		return lExpectedWeight;
	}
}
