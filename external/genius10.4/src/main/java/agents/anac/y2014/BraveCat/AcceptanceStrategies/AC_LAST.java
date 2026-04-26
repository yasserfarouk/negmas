package agents.anac.y2014.BraveCat.AcceptanceStrategies;

import genius.core.boaframework.Actions;

public class AC_LAST extends AcceptanceStrategy
{
    @Override
    public Actions determineAcceptability()
    {
        try
        {
            double MyBestBidUtility = this.negotiationSession.getUtilitySpace().getUtility(this.negotiationSession.getOwnBidHistory().getBestBidDetails().getBid());
            double MyWorstBidUtility = this.negotiationSession.getUtilitySpace().getUtility(this.negotiationSession.getOwnBidHistory().getWorstBidDetails().getBid());
            double OpponentLastBidUtility = this.negotiationSession.getUtilitySpace().getUtility(this.negotiationSession.getOpponentBidHistory().getLastBid());
            
            if(OpponentLastBidUtility < this.negotiationSession.getUtilitySpace().getReservationValue())
                return Actions.Reject;
            //-------------------------------------------------------------------------------------------------------------------------------
            //Tactic #1.
            if(OpponentLastBidUtility >= 0.8)
                return Actions.Accept;
            //Tactic #2.
            if(OpponentLastBidUtility >= MyBestBidUtility)
                return Actions.Accept;
            //Tactic #3.
            if(OpponentLastBidUtility >= (double) (MyBestBidUtility + MyWorstBidUtility) / 2)
                return Actions.Accept;
            //Tactic #4.
            if(this.schedular.LastRound() || this.schedular.FinalRounds())
                return Actions.Accept;
            //-------------------------------------------------------------------------------------------------------------------------------
            return Actions.Reject;
        }
        catch (Exception ex)
        {
            System.out.println("Exception occurred while determining acceptability!");
        }
        return Actions.Reject;
    }
}