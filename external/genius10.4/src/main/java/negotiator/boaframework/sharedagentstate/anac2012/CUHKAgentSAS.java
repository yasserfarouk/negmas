package negotiator.boaframework.sharedagentstate.anac2012;

import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;
import genius.core.misc.Queue;

public class CUHKAgentSAS extends SharedAgentState{
	
	private NegotiationSession negotiationSession;
    private double timeLeftBefore;
    private double timeLeftAfter;
    private double maximumTimeOfOpponent;
    private double maximumTimeOfOwn;
    private double totalTime;
    private boolean concedeToOpponent;
    private boolean toughAgent; //if we propose a bid that was proposed by the opponnet, then it should be accepted.
    private double utilitythreshold;
    private double MaximumUtility;
    private double concedeToDiscountingFactor;
	private Queue timeInterval = new Queue(5);

	public CUHKAgentSAS (NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "CUHKAgent";	
        this.timeLeftBefore = 0;
        this.timeLeftAfter = 0;
        this.maximumTimeOfOpponent = 0;
        this.maximumTimeOfOwn = 0;
        this.totalTime = negoSession.getTimeline().getTotalTime();
        this.concedeToOpponent = false;
        this.toughAgent = false;
        try {
			this.utilitythreshold = negoSession.getMaxBidinDomain().getMyUndiscountedUtil();
		} catch (Exception e) {
			e.printStackTrace();
		}//initial utility threshold
        this.MaximumUtility = this.utilitythreshold;
	}
	public int estimateTheRoundsLeft(boolean activeHelper, boolean opponent) {
		int roundsLeft = 0;
		if(!activeHelper){
			estimateRoundLeft(opponent);
		}else {
			estimateRoundsNoActiveHelper();
		}
		
		return roundsLeft;
	}
	
	private int estimateRoundsNoActiveHelper(){
		int roundsLeft = 0;
		Double[] array = timeInterval.toArray();
		double total = 0;

		for (double i : array){
			total += i;
		}
		//determine the average time per round
		double average = total/timeInterval.size();
		double timeLeft = negotiationSession.getTimeline().getTotalTime() - negotiationSession.getTime();
		roundsLeft = (int) (timeLeft/average);
		
		return roundsLeft;
	}
	
	/*
     * estimate the number of rounds left before reaching the deadline @param
     * opponent @return
     */
	private int estimateRoundLeft(boolean opponent) {
        double round;
        if (opponent == true) {
            if (this.timeLeftBefore - this.timeLeftAfter > this.maximumTimeOfOpponent) {
                this.maximumTimeOfOpponent = this.timeLeftBefore - this.timeLeftAfter;
            }
        } else {
            if (this.timeLeftAfter - this.timeLeftBefore > this.maximumTimeOfOwn) {
                this.maximumTimeOfOwn = this.timeLeftAfter - this.timeLeftBefore;
            }
        }
        if (this.maximumTimeOfOpponent + this.maximumTimeOfOwn == 0) {
            System.out.println("divided by zero exception");
        }
        round = (this.totalTime - negotiationSession.getTimeline().getCurrentTime()) / (this.maximumTimeOfOpponent + this.maximumTimeOfOwn);
        //System.out.println("current time is " + timeline.getElapsedSeconds() + "---" + round + "----" + this.maximumTimeOfOpponent);
        return ((int) (round));
    }

	public void setTimeLeftBefore(double timeLeftBefore) {
		this.timeLeftBefore = timeLeftBefore;
	}

	public void setTimeLeftAfter(double timeLeftAfter) {
		this.timeLeftAfter = timeLeftAfter;
	}

	public double getMaximumTimeOfOpponent() {
		return maximumTimeOfOpponent;
	}

	public void setMaximumTimeOfOpponent(double maximumTimeOfOpponent) {
		this.maximumTimeOfOpponent = maximumTimeOfOpponent;
	}

	public double getMaximumTimeOfOwn() {
		return maximumTimeOfOwn;
	}

	public void setMaximumTimeOfOwn(double maximumTimeOfOwn) {
		this.maximumTimeOfOwn = maximumTimeOfOwn;
	}
    public boolean isConcedeToOpponent() {
		return concedeToOpponent;
	}

	public void setConcedeToOpponent(boolean concedeToOpponent) {
		this.concedeToOpponent = concedeToOpponent;
	}

	public boolean isToughAgent() {
		return toughAgent;
	}

	public void setToughAgent(boolean toughAgent) {
		this.toughAgent = toughAgent;
	}

	public double getUtilitythreshold() {
		return utilitythreshold;
	}

	public void setUtilitythreshold(double utilitythreshold) {
		this.utilitythreshold = utilitythreshold;
	}

	public double getMaximumUtility() {
		return MaximumUtility;
	}

	public double getConcedeToDiscountingFactor() {
		return concedeToDiscountingFactor;
	}

	public void setConcedeToDiscountingFactor(double concedeToDiscountingFactor) {
		this.concedeToDiscountingFactor = concedeToDiscountingFactor;
	}	
	
	public void addTimeInterval(double time){
		timeInterval.enqueue(time);
		timeInterval.dequeue();
	}

}
