package agents.anac.y2018.yeela;

import java.util.List;

import java.util.HashMap;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

public class Individual implements Comparable<Individual> {
	private Double ALPHA = 0.9; // TODO
	
	private HashMap<Integer, Value> m_gene;
	private Double m_util;
	private NegotiationInfo m_info;
	private Random m_rand;
	private Bid m_opponent;
	private Double m_best;
	private Double m_maxDist;
	
	public Individual(Bid b, NegotiationInfo info)
	{
		m_gene = b.getValues();
		m_info = info;
		m_util = CalcUtility();
		m_rand = new Random();
		
        try
        {
        	Bid best = m_info.getUtilitySpace().getMaxUtilityBid();
        	m_best = m_info.getUtilitySpace().getUtility(best);
        	m_maxDist = Dist(
        		best.getValues(),
				m_info.getUtilitySpace().getMinUtilityBid().getValues());
        }
        catch (Exception e)
        {
        	m_maxDist = Double.MIN_VALUE;
            e.printStackTrace();
        }
	}
	
	@Override
	public int compareTo(Individual other)
	{
    	try
    	{
    		return (int) java.lang.Math.signum(this.GetFitness() - other.GetFitness());
 		}
    	catch (Exception e)
    	{
			e.printStackTrace();
			return 0;
		}
	}
	
	public Value GetValue(Integer key)
	{
		return m_gene.get(key);
	}
	
	public void SetValue(Integer key, Value value)
	{
		m_gene.put(key, value);
	}
	
	private Double TP()
	{
		// TODO
		return 1.0;
	}
	
	private Double Dist(HashMap<Integer, Value> v1, HashMap<Integer, Value> v2)
	{
		Double d = 0.0;
		
		AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) m_info.getUtilitySpace();
		List<Issue> issues = additiveUtilitySpace.getDomain().getIssues();

		for (Issue issue : issues)
		{
		    int issueNumber = issue.getNumber();
		    Double weight = additiveUtilitySpace.getWeight(issueNumber);

		    // Assuming that issues are discrete only
		    EvaluatorDiscrete evaluatorDiscrete = (EvaluatorDiscrete) additiveUtilitySpace.getEvaluator(issueNumber);
		    
        	try
        	{
    		    Double evaluation1 = evaluatorDiscrete.getEvaluation((ValueDiscrete)v1.get(issueNumber));
    		    Double evaluation2 = evaluatorDiscrete.getEvaluation((ValueDiscrete)v2.get(issueNumber));
			
				d += (weight * Math.pow(evaluation1 - evaluation2, 2));
			}
        	catch (Exception e)
        	{
        		d = Double.POSITIVE_INFINITY;
				e.printStackTrace();
			}
		}
		
		return Math.sqrt(d);
	}
	
	public void UpdateOpponent(Bid opponent)
	{
		m_opponent = opponent;
	}
	
	public Double GetFitness()
	{
        Double otherSide = (1 - ALPHA) * TP() * (1 - (Dist(m_gene, m_opponent.getValues()) / m_maxDist));
        Double ourSide = ALPHA * TP() * (this.GetUtility() / m_best);
		return ourSide + otherSide;
	}
	
	public Double GetUtility()
	{
		return m_util;
	}
	
	private Double CalcUtility()
	{
		// TODO maybe simply call utilitySpace.getUtility(randomBid); see https://github.com/tdgunes/ExampleAgent/wiki/Generating-a-random-bid-with-a-utility-threshold
		Double util = 0.0;

		AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) m_info.getUtilitySpace();
		List<Issue> issues = additiveUtilitySpace.getDomain().getIssues();

		for (Issue issue : issues)
		{
		    int issueNumber = issue.getNumber();
		    Double weight = additiveUtilitySpace.getWeight(issueNumber);

		    // Assuming that issues are discrete only
		    IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
		    EvaluatorDiscrete evaluatorDiscrete = (EvaluatorDiscrete) additiveUtilitySpace.getEvaluator(issueNumber);
		    Double evaluation = 0.0;
		    
		    for (ValueDiscrete valueDiscrete : issueDiscrete.getValues())
		    {
		        if (0 == valueDiscrete.getValue().compareTo(m_gene.get(issueNumber).toString()))
		        {
		        	try
		        	{
						evaluation = evaluatorDiscrete.getEvaluation(valueDiscrete);
						util += (weight * evaluation);
					}
		        	catch (Exception e)
		        	{
						e.printStackTrace();
					}
		        }
		    }
		}
		return util;
	}
	
	public Individual Clone()
	{
		Bid randomBid = m_info.getUtilitySpace().getDomain().getRandomBid(m_rand);

    	try
    	{
    		for (Integer key : m_gene.keySet())
    		{
    			randomBid.putValue(key, m_gene.get(key));
    		}
		}
    	catch (Exception e)
    	{
			e.printStackTrace();
		}
		
		Individual ind = new Individual(randomBid, m_info);
		ind.UpdateOpponent(m_opponent);
		return ind;
	}
	
	public void Mutate()
	{
		try
		{
			AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) m_info.getUtilitySpace();
			List<Issue> issues = additiveUtilitySpace.getDomain().getIssues();
			
			int randomNum = m_rand.nextInt(issues.size());
			Issue issue = issues.get(randomNum);
			
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			int issueNumber = issue.getNumber();
			randomNum = issueNumber;
			while (randomNum == issueNumber)
			{
				randomNum = m_rand.nextInt(issueDiscrete.getValues().size());
			}
			
			Value newValue = issueDiscrete.getValues().get(randomNum);
			m_gene.put(issueNumber, newValue);
		}
    	catch (Exception e)
    	{
			e.printStackTrace();
		}

	}
	
	public void Crossover(Individual other)
	{
		try
		{
			AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) m_info.getUtilitySpace();
			List<Issue> issues = additiveUtilitySpace.getDomain().getIssues();
			
			if (1 < issues.size()) // otherwise no need to crossover
			{
				int randomNum = m_rand.nextInt(issues.size() - 1) + 1; // minus one since crossover location is in gap between alleles
				
				for (Integer key : m_gene.keySet())
				{
					if (key > randomNum)
					{	
						break;
					}
					
					Value temp = m_gene.get(key);
					m_gene.put(key, other.GetValue(key));
					other.SetValue(key, temp);
				}
			}
		}
    	catch (Exception e)
    	{
			e.printStackTrace();
		}
	}
}
