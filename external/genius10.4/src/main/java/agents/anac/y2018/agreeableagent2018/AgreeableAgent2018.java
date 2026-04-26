package agents.anac.y2018.agreeableagent2018;

import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.OutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

/**
 * Created by Sahar Mirzayi
 * 2/14/2018
 * University of Tehran
 * Agent Lab.
 * Sahar.Mirzayi @ gmail.com
 */

public class AgreeableAgent2018 extends AbstractNegotiationParty {
    private int domainSize;
    private double pMin;
    private double pMax;
    private AbstractUtilitySpace utilSpace = null;
    private TimeLineInfo timeLineInfo = null;
    private Bid bestBid = null;
    private Bid lastReceivedOffer = null;
    private OutcomeSpace outcomeSpace = null;
    private FrequencyBasedOpponentModel opponentModel1;
    private FrequencyBasedOpponentModel opponentModel2;
    private AgentID opponent1ID = null;
    private AgentID opponent2ID = null;
    protected int opponent1BidCount = 0;
    protected int opponent2BidCount = 0;
    private double timeForUsingModel = 0.1;
    private boolean canUseModel = false;


    @Override
    public void init(NegotiationInfo info) {
        super.init(info);
        this.utilSpace = info.getUtilitySpace();
        outcomeSpace = new OutcomeSpace(utilSpace);
        domainSize = outcomeSpace.getAllOutcomes().size();

        timeLineInfo = info.getTimeline();
        try {
            bestBid = info.getUtilitySpace().getMaxUtilityBid();
            pMin = utilitySpace.getUtility(utilitySpace.getMinUtilityBid());
            pMax = utilitySpace.getUtility(utilitySpace.getMaxUtilityBid());
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (isAllIssuesDiscrete()) {
            canUseModel = true;
            determineTimeForUsingModel();
            opponentModel1 = new FrequencyBasedOpponentModel();
            opponentModel1.init(utilitySpace.getDomain().getIssues());
            opponentModel2 = new FrequencyBasedOpponentModel();
            opponentModel2.init(utilitySpace.getDomain().getIssues());
        } else {
            canUseModel = false;
        }
    }

    private boolean isAllIssuesDiscrete() {
        List<Issue> issues = utilSpace.getDomain().getIssues();
        for (Issue issue : issues) {
            if (!(issue instanceof IssueDiscrete))
                return false;
        }
        return true;
    }

    private void determineTimeForUsingModel() {
        if (domainSize < Constants.smallDomainUpperBound)
            timeForUsingModel = Constants.timeForUsingModelForSmallDomain;
        if (domainSize >= Constants.smallDomainUpperBound &&
                domainSize < Constants.midDomainUpperBound)
            timeForUsingModel = Constants.timeForUsingModelForMidDomain;
        if (domainSize >= Constants.midDomainUpperBound)
            timeForUsingModel = Constants.timeForUsingModelForLargeDomain;
    }

    @Override
    public void receiveMessage(AgentID sender, Action lastOpponentAction) {
        super.receiveMessage(sender, lastOpponentAction);
        if (lastOpponentAction instanceof Offer) {
            Bid bid = ((Offer) lastOpponentAction).getBid();
            if (canUseModel) {
                if (opponent1ID == null) {
                    opponent1ID = sender;
                } else if (opponent2ID == null) {
                    opponent2ID = sender;
                }
                if (opponent1ID == sender) {
                    opponent1BidCount++;
                    opponentModel1.updateModel(bid, opponent1BidCount);
                }
                if (opponent2ID == sender) {
                    opponent2BidCount++;
                    opponentModel2.updateModel(bid, opponent2BidCount);
                }
            }
            lastReceivedOffer = ((Offer) lastOpponentAction).getBid();
        }
    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        if (lastReceivedOffer != null) {
            try {
                Bid myBid = getNextBid();
                if (isAcceptable(getUtility(lastReceivedOffer), getUtility(myBid)))
                    return new Accept(getPartyId(), lastReceivedOffer);
                else
                    return new Offer(getPartyId(), myBid);
            } catch (Exception ex) {
                return new Offer(getPartyId(), bestBid);
            }
        } else {
            return new Offer(getPartyId(), bestBid);
        }
    }

    private boolean isAcceptable(double opponentUtility, double myBidUtilByTime) {
        if (opponentUtility >= myBidUtilByTime)
            return true;

        double time = timeLineInfo.getTime();
        return time >= 0.99 && opponentUtility >= utilSpace.getReservationValue();

    }

    private Bid getNextBid() {

        double time = timeLineInfo.getTime();
        double targetUtilityByTime = getUtilityByTime(time);
        if (targetUtilityByTime < Constants.minimumUtility)
            targetUtilityByTime = Constants.minimumUtility;

        if (isModelUsable() && canUseModel) {
            return tuneBidByOpponentModel(targetUtilityByTime);
        } else {
            BidDetails bidNearUtility = outcomeSpace.getBidNearUtility(targetUtilityByTime);
            return bidNearUtility.getBid();
        }
    }

    private boolean isModelUsable() {
        double time = timeLineInfo.getTime();
        return time >= timeForUsingModel;
    }

    private Bid tuneBidByOpponentModel(double targetUtilityByTime) {

        double utilityThreshold = getExplorableNeighbourhood();
        Range range =
                new Range(targetUtilityByTime - utilityThreshold, targetUtilityByTime + utilityThreshold);
        List<BidDetails> bidsInRange = outcomeSpace.getBidsinRange(range);
        if (bidsInRange.size() == 1)
            return bidsInRange.get(0).getBid();
        int selectedBidIndex = getBestBidByRouletteWheel(bidsInRange);
        return bidsInRange.get(selectedBidIndex).getBid();
    }


    /*private int getBestBidByOpponentUtilities(List<BidDetails> bidsInRange) {
        int size = bidsInRange.size();
        double max = 0;
        int maxIndex = 0;
        for (int i = 0; i < size; i++) {
            BidDetails bidDetails = bidsInRange.get(i);
            double sum = opponentModel1.getUtility(bidDetails.getBid())
                    + opponentModel2.getUtility(bidDetails.getBid());
            if (sum > max) {
                max = sum;
                maxIndex = i;
            }
        }
        return maxIndex;

    }*/

    private int getBestBidByRouletteWheel(List<BidDetails> bidsInRange) {
        int size = bidsInRange.size();
        double[] sumOfTwoUtilitiesForBid = new double[size];
        double totalUtility = 0;
        for (int i = 0; i < size; i++) {
            BidDetails bidDetails = bidsInRange.get(i);
            double sum = opponentModel1.getUtility(bidDetails.getBid())
                    + opponentModel2.getUtility(bidDetails.getBid());
            sumOfTwoUtilitiesForBid[i] = sum;
            totalUtility += sum;
        }
        double[] normalizedSumOfTwoUtilitiesForBid = new double[size];
        for (int i = 0; i < size; i++) {
            normalizedSumOfTwoUtilitiesForBid[i] = sumOfTwoUtilitiesForBid[i] / totalUtility;
        }
        double random = Math.random();
        double integrate = 0;
        int selectedBidIndex = size;
        for (int i = 0; i < size; i++) {
            integrate += normalizedSumOfTwoUtilitiesForBid[i];
            if (integrate >= random) {
                selectedBidIndex = i;
                break;
            }
        }
        return selectedBidIndex;
    }

    private double getExplorableNeighbourhood() {
        double time = timeLineInfo.getTime();
        if (time < Constants.timeToConcede) {
            return 0;
        } else {
             return Constants.neigExplorationDisFactor
                     *(1 - (pMin + (pMax - pMin) * (1 - f(time))));
        }
    }

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

    public double getUtilityByTime(double time) {
        if (time < Constants.timeToConcede) {
            return 1;
        } else {
            time = (time - Constants.timeToConcede) / (1 - Constants.timeToConcede);//normalization
            return pMin + (pMax - pMin) * (1 - f(time));
        }
    }

    public double f(double t) {
        if (Constants.concessionFactor == 0)
            return Constants.k;
        return Constants.k + (1 - Constants.k) * Math.pow(t, 1.0 / Constants.concessionFactor);
    }
}
