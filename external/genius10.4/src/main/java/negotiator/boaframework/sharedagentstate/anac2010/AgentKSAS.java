package negotiator.boaframework.sharedagentstate.anac2010;

import java.util.HashMap;
import java.util.Random;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC 2010 AgentFSEGA.
 * The code was taken from the ANAC2010 AgentFSEGA and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class AgentKSAS extends SharedAgentState{

	private NegotiationSession negotiationSession;
	private double sum;
	private double sum2;
	private int rounds;
	private double target;
	private double bidTarget;
	private double tremor;
	private double p;
	private Random random400;
	private HashMap<Bid, Double> offeredBidMap;
	private final boolean TEST_EQUIVALENCE = false;
	
	public AgentKSAS (NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "AgentK";
		offeredBidMap = new HashMap<Bid, Double>();
		sum = 0.0;
		sum2 = 0.0;
		rounds = 0;
        target = 1.0;
        bidTarget = 1.0;
        tremor = 2.0;
        if (TEST_EQUIVALENCE) {
        	random400 = new Random(400);
        } else {
        	random400 = new Random();
        }
	}
	
    public double calculateAcceptProbability() {

        double offeredUtility = negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil();
        offeredBidMap.put(negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid(), offeredUtility);
        
        sum += offeredUtility;
        sum2 += offeredUtility * offeredUtility;
        rounds++;
        
        double mean = sum / rounds;

        double variance = (sum2 / rounds) - (mean * mean);

        double deviation = Math.sqrt(variance * 12);
        if (Double.isNaN(deviation)) {
            deviation = 0.0;
        }

        //double time = ((new Date()).getTime() - startTime.getTime()) // get passed time in ms
        //        / (1000. * totalTime); // divide by 1000 * totalTime to get normalized time between 0 and 1
        double time = negotiationSession.getTime();
        double t = time * time * time;

        if (offeredUtility > 1.) {
            offeredUtility = 1;
        }
        double estimateMax = mean + ((1 - mean) * deviation);

        double alpha = 1 + tremor + (10 * mean) - (2 * tremor * mean);
        double beta = alpha + (random400.nextDouble() * tremor) - (tremor / 2);

        double preTarget = 1 - (Math.pow(time, alpha) * (1 - estimateMax));
        double preTarget2 = 1 - (Math.pow(time, beta) * (1 - estimateMax));

        double ratio = (deviation + 0.1) / (1 - preTarget);
        if (Double.isNaN(ratio) || ratio > 2.0) {
            ratio = 2.0;
        }

        double ratio2 = (deviation + 0.1) / (1 - preTarget2);
        if (Double.isNaN(ratio2) || ratio2 > 2.0) {
            ratio2 = 2.0;
        }

        target = ratio * preTarget + 1 - ratio;
        bidTarget = ratio2 * preTarget2 + 1 - ratio2;

        double m = t * (-300) + 400;
        if (target > estimateMax) {
            double r = target - estimateMax;
            double f = 1 / (r * r);
            if (f > m || Double.isNaN(f))
                f = m;
            double app = r * f / m;
            target = target - app;
        } else {
            target = estimateMax;
        }

        if (bidTarget > estimateMax) {
            double r = bidTarget - estimateMax;
            double f = 1 / (r * r);
            if (f > m || Double.isNaN(f))
                f = m;
            double app = r * f / m;
            bidTarget = bidTarget - app;
        } else {
            bidTarget = estimateMax;
        }

        double utilityEvaluation = offeredUtility - estimateMax;
        double satisfy = offeredUtility - target;

        p = (Math.pow(time, alpha) / 5) + utilityEvaluation + satisfy;
        if (p < 0.1) {
            p = 0.0;
        }
        return p;
    }
    
    public double getTarget() {
    	return target;
    }
    
    public double getBidTarget() {
    	return bidTarget;
    }

	public void decrementBidTarget(double d) {
		bidTarget = bidTarget - d;
	}
	
	public int getRounds() {
		return rounds;
	}
	
	public HashMap<Bid, Double> getOfferedBidMap() {
		return offeredBidMap;
	}

	public double getAcceptProbability() {
		return p;
	}
}