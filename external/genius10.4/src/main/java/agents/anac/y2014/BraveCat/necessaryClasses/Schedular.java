package agents.anac.y2014.BraveCat.necessaryClasses;

import java.util.ArrayList;
import java.util.List;

public class Schedular 
{
    private List<Double> ReceivedBidsDurations;
    private List<Double> SentBidsDurations;
    private NegotiationSession negotiationSession;
    
    public Schedular(NegotiationSession negoSession) throws Exception
    {
        negotiationSession = negoSession;
        ReceivedBidsDurations = new ArrayList();
        SentBidsDurations = new ArrayList();
    }
    public Boolean LastRound()
    {
        double LastReceivedBidsDuration = 0;
        double ExpectedLastBidTime = 0;
        int size = this.negotiationSession.getOpponentBidHistory().getHistory().size();
        if(size >= 2)
        {
            LastReceivedBidsDuration = this.negotiationSession.getOpponentBidHistory().getHistory().get(size - 1).getTime() - this.negotiationSession.getOpponentBidHistory().getHistory().get(size - 2).getTime();
            ReceivedBidsDurations.add(LastReceivedBidsDuration);
            if(size >= 3)
                LastReceivedBidsDuration = Math.min(ReceivedBidsDurations.get(ReceivedBidsDurations.size() - 1), ReceivedBidsDurations.get(ReceivedBidsDurations.size() - 2));
            if(size >= 4)
                LastReceivedBidsDuration = Math.min(LastReceivedBidsDuration, ReceivedBidsDurations.get(ReceivedBidsDurations.size() - 3));
            
            if(size * 0.002 >= 1)
                LastReceivedBidsDuration = size * 0.002 * LastReceivedBidsDuration;
            
            ExpectedLastBidTime = this.negotiationSession.getOpponentBidHistory().getLastBidDetails().getTime() + (2 * LastReceivedBidsDuration);
            
            //System.out.println(ExpectedLastBidTime);

            if(ExpectedLastBidTime > 1)
            {
                System.out.println("Final Rounds (1): " + this.negotiationSession.getTime()  + " ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
                return true;
            }
        }
        return false;        
    }
    public Boolean FinalRounds()
    {
        double LastSentBidsDuration = 0;
        double ExpectedLastBidTime = 0;
        int size = this.negotiationSession.getOwnBidHistory().getHistory().size();
        if(size >= 1)
        {
            LastSentBidsDuration = negotiationSession.getTime() - negotiationSession.getOwnBidHistory().getLastBidDetails().getTime();
            SentBidsDurations.add(LastSentBidsDuration);
            if(size >= 2)
                LastSentBidsDuration = Math.min(SentBidsDurations.get(SentBidsDurations.size() - 1), SentBidsDurations.get(SentBidsDurations.size() - 2));
            if(size >= 3)
                LastSentBidsDuration = Math.min(LastSentBidsDuration, SentBidsDurations.get(SentBidsDurations.size() - 3));
            
            if(size * 0.002 >= 1)
                LastSentBidsDuration = size * 0.002 * LastSentBidsDuration;
            
            ExpectedLastBidTime = negotiationSession.getTime() + (2 * LastSentBidsDuration);
            
            //System.out.println(ExpectedLastBidTime);
            
            if(ExpectedLastBidTime > 1)
            {
                System.out.println("Final Rounds (2): " + this.negotiationSession.getTime() + " ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
                return true;
            }
        }
        return false;
    }
}
