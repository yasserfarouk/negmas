package agents.anac.y2018.yeela;

import java.util.List;

import java.util.Collection;
import java.util.Collections;
import java.util.Random;
import java.util.Vector;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.parties.NegotiationInfo;


public class Learner {
	private int MAX_EVOLUTIONS = 10; //TODO
	private int POPULATION_SIZE = 200; // TODO
	private int MATING_POOL_SIZE = 150; // TODO
	private double CROSS_RATE = 0.6; // TODO
	private double MUT_RATE = 0.05; // TODO
	private double ELITISM_RATE = 0.1; // TODO
	
	private List<Individual> m_population;
	private Random m_rand;
	private NegotiationInfo m_info;
			
	public Learner(Bid bestBid, NegotiationInfo info)
	{
		m_info = info;
		
		m_population = new Vector<Individual>();
		m_population.add(new Individual(bestBid, info));
		
		// generate initial population
		m_rand = new Random();
		for (int i = 0; i < POPULATION_SIZE - 1; ++i)
		{
			Bid randomBid = info.getUtilitySpace().getDomain().getRandomBid(m_rand);
			m_population.add(new Individual(randomBid, info));
		}
	}
	
	private List<Individual> Best()
	{
		Vector<Individual> out = new Vector<Individual>();
		
		try
		{
			Collections.sort(m_population);
			
			int popSize = m_population.size();
			int start = (int)(popSize - (popSize * ELITISM_RATE));
			int end = popSize;
			
			for (Individual ind : m_population.subList(start, end))
			{
				out.add(ind.Clone());
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
		return out;
	}
	
	private List<Individual> Selection()
	{
		List<Individual> matingPool = new Vector<Individual>();
		int popSize = m_population.size();
		try
		{
			Collections.sort(m_population);
			boolean[] taken = new boolean[popSize];
			
			while (matingPool.size() < MATING_POOL_SIZE)
			{
				for (int i = 0; i < popSize; ++i)
				{
					if(!taken[i])
					{
						double pr = m_rand.nextDouble();
						if ((1.0 / (i + 1)) > pr)
						{
							matingPool.add(m_population.get(popSize - 1 - i).Clone());
							taken[i] =  true;
						}
					}
				}
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
			matingPool = m_population.subList(0, (int)(m_population.size() * 0.75));
		}

		
		return matingPool;
	}
	
	public Bid run(Bid counterOffer)
	{
		m_population.forEach(ind->ind.UpdateOpponent(counterOffer));
		Collection<Individual> new_generation;
		for (int i = 0; i < MAX_EVOLUTIONS; i++)
		{
			new_generation = Best();
			List<Individual> mating_pool = Selection();
			while(new_generation.size() < POPULATION_SIZE)
			{
				for (int j = 0; j < mating_pool.size() - 1; j += 2)
				{
					Individual ind = mating_pool.get(j);
					Individual ind2 = mating_pool.get(j+1);
					if (m_rand.nextDouble() > CROSS_RATE)
					{
						ind.Crossover(ind2);					
					}
					if (m_rand.nextDouble() > MUT_RATE)
					{
						ind.Mutate();
					}
					if (m_rand.nextDouble() > MUT_RATE)
					{
						ind2.Mutate();
					}
					new_generation.add(ind);
					new_generation.add(ind2);
				}
			}
			m_population = new Vector<Individual>(new_generation);
		}
		
		Bid answer = m_info.getUtilitySpace().getDomain().getRandomBid(m_rand);
		Individual ind = Best().get(0);
		
		for (Issue issue : answer.getIssues())
		{
			answer.putValue(issue.getNumber(), ind.GetValue(issue.getNumber()));
		}
		
		return answer;
	}
}
